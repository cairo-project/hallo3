"""Inference wrapper for Hallo3 exposing a load-once / generate-per-clip API.

This adapts the upstream ``sample_video.py`` (a SAT, config-driven batch script)
into two reusable functions so a long-running worker can load the model once and
generate many clips:

    models = load_models(model_dir, s2_config, inference_config)
    generate_video(models, source_image, driving_audio, save_path, prompt="")

The per-clip inference logic is taken verbatim from ``sample_video.sampling_main``
(its helper functions are imported, not copied, so this stays in lock-step with
upstream). Only the model/processor setup and the single-item driver are new.

NOTE: Hallo3 is SAT + CogVideoX-5B based and needs the full weight set plus a
CUDA GPU + the SAT/deepspeed stack to run; this wrapper is authored from the
upstream source and has not been executed end-to-end here. Validate on a GPU
host with the weights present before relying on it.
"""

import argparse
import math
import os
import shutil
import tempfile

import torch

# Helpers reused verbatim from the upstream batch script.
from sample_video import (
    add_mask_to_first_frame,
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    process_audio_emb,
    resize_for_rectangle_crop,
    resize_for_square_padding,
    save_video_as_grid_and_mp4_with_audio,
)

_PRETRAINED_TOKEN = "./pretrained_models"


def _single_gpu_env() -> None:
    """SAT/mpu reads these at import/init; default them for single-process use."""
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29555")


def _patch_config(src_path: str, model_dir: str, out_dir: str) -> str:
    """Copy a config, repointing ``./pretrained_models`` at ``model_dir``."""
    with open(src_path) as f:
        text = f.read()
    text = text.replace(_PRETRAINED_TOKEN, model_dir.rstrip("/"))
    out_path = os.path.join(out_dir, os.path.basename(src_path))
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def load_models(
    model_dir: str,
    s2_config: str,
    inference_config: str,
    seed: int = 42,
):
    """Build the Hallo3 SAT engine + audio/image processors.

    Args:
        model_dir:         root of the downloaded weights (the HF repo layout:
                           hallo3/, cogvideox-5b-i2v-sat/, t5-v1_1-xxl/, wav2vec/,
                           face_analysis/, audio_separator/).
        s2_config:         path to cogvideox_5b_i2v_s2.yaml (model definition).
        inference_config:  path to inference.yaml (paths + sampling settings).
        seed:              base RNG seed.

    Returns a dict consumed by ``generate_video``.
    """
    _single_gpu_env()

    # Patch the two configs so every ./pretrained_models/* path points at model_dir.
    cfg_dir = tempfile.mkdtemp(prefix="hallo3_cfg_")
    s2_patched = _patch_config(s2_config, model_dir, cfg_dir)
    inf_patched = _patch_config(inference_config, model_dir, cfg_dir)

    # Import here so the single-GPU env is set first.
    from sat.model.base_model import get_model
    from sat.training.model_io import load_checkpoint
    from arguments import get_args
    from diffusion_video import SATVideoDiffusionEngine
    from sgm.utils.audio_processor import AudioProcessor
    from sgm.utils.image_processor import ImageProcessor

    args = get_args(["--base", s2_patched, inf_patched,
                     "--input-type", "txt", "--seed", str(seed)])
    args = argparse.Namespace(**vars(args))

    # Same single-GPU inference overrides as sample_video.__main__.
    if hasattr(args, "deepspeed_config"):
        del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    model = get_model(args, SATVideoDiffusionEngine)
    load_checkpoint(model, args)
    model.eval()
    model = model.to("cuda")

    audio_processor = AudioProcessor(
        args.sample_rate,
        args.wav2vec_model_path,
        args.wav2vec_features == "last",
        os.path.dirname(args.audio_separator_model_path),
        os.path.basename(args.audio_separator_model_path),
        os.path.join(".cache", "audio_preprocess"),
    )
    image_processor = ImageProcessor(args.face_analysis_model_path)

    return {
        "model": model,
        "audio_processor": audio_processor,
        "image_processor": image_processor,
        "args": args,
    }


@torch.no_grad()
def generate_video(
    models: dict,
    source_image: str,
    driving_audio: str,
    save_path: str,
    prompt: str = "",
    face_expand_ratio: float = 1.2,
    seed: int = 42,
) -> str:
    """Generate one audio-driven clip and write it to ``save_path`` (mp4)."""
    from PIL import Image
    import torchvision.transforms as TT

    model = models["model"]
    audio_processor = models["audio_processor"]
    image_processor = models["image_processor"]
    args = models["args"]

    image_size = [480, 720]
    T, C, F = args.sampling_num_frames, args.latent_channels, 8
    H, W = image_size
    L = (T - 1) * 4 + 1
    n_motion_frame = 2
    mask_rate = 0.1
    num_samples = [1]
    device = model.device
    transform = TT.Compose([TT.ToTensor()])

    work_dir = tempfile.mkdtemp(prefix="hallo3_out_")
    try:
        audio_emb, length = audio_processor.preprocess(driving_audio, L)
        audio_emb = process_audio_emb(audio_emb)

        face_emb, face_mask_path = image_processor.preprocess(source_image, work_dir, face_expand_ratio)
        face_emb = torch.tensor(face_emb.reshape(1, -1)).to("cuda")

        image = transform(Image.open(source_image).convert("RGB")).unsqueeze(0).to("cuda")
        face_mask = transform(Image.open(face_mask_path).convert("RGB")).unsqueeze(0).to("cuda")
        ref_image = image * face_mask

        _, _, h, w = image.shape
        is_padding = h == w

        if is_padding:
            image = resize_for_square_padding(image, image_size).clamp(0, 1)
            ref_image = resize_for_square_padding(ref_image, image_size).clamp(0, 1)
        else:
            image = resize_for_rectangle_crop(image, image_size, reshape_mode="center").unsqueeze(0)
            ref_image = resize_for_rectangle_crop(ref_image, image_size, reshape_mode="center").unsqueeze(0)

        image = image * 2.0 - 1.0
        motion_image = image.unsqueeze(2).to(torch.bfloat16)
        ref_image_pixel = image.unsqueeze(2).to(torch.bfloat16)

        ref_image = (ref_image * 2.0 - 1.0).unsqueeze(2).to(torch.bfloat16)

        motion_image = torch.cat([motion_image] * n_motion_frame, dim=2)
        mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
        mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
        mask_image = model.encode_first_stage(mask_image, None).permute(0, 2, 1, 3, 4).contiguous()

        ref_image = model.encode_first_stage(ref_image, None).permute(0, 2, 1, 3, 4).contiguous()

        pad_shape = (mask_image.shape[0], T - 1, C, H // F, W // F)
        mask_image = torch.concat(
            [mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1
        )

        value_dict = {"prompt": prompt, "negative_prompt": "",
                      "num_frames": torch.tensor(T).unsqueeze(0)}
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
        )
        c, uc = model.conditioner.get_unconditional_conditioning(
            batch, batch_uc=batch_uc, force_uc_zero_embeddings=["txt"]
        )
        for k in c:
            if k != "crossattn":
                c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

        times = audio_emb.shape[0] // (L - n_motion_frame)
        if times * (L - n_motion_frame) < audio_emb.shape[0]:
            times += 1
        video = []
        pre_fix = torch.zeros_like(audio_emb[:n_motion_frame])

        for t in range(times):
            c["concat"] = mask_image
            uc["concat"] = mask_image
            audio_tensor = audio_emb[
                t * (L - n_motion_frame): min((t + 1) * (L - n_motion_frame), audio_emb.shape[0])
            ]
            audio_tensor = torch.cat([pre_fix, audio_tensor], dim=0)
            pre_fix = audio_tensor[-n_motion_frame:]
            if audio_tensor.shape[0] != L:
                pad = L - audio_tensor.shape[0]
                padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                audio_tensor = torch.cat([audio_tensor, padding], dim=0)
            audio_tensor = audio_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            samples_z = model.sample(
                c, uc=uc, batch_size=1, shape=(T, C, H // F, W // F),
                audio_emb=audio_tensor, ref_image=ref_image, face_emb=face_emb,
            ).permute(0, 2, 1, 3, 4).contiguous()
            torch.cuda.empty_cache()

            latent = 1.0 / model.scale_factor * samples_z
            recons = []
            loop_num = (T - 1) // 2
            for i in range(loop_num):
                start_frame, end_frame = (0, 3) if i == 0 else (i * 2 + 1, i * 2 + 3)
                clear_fake_cp_cache = i == loop_num - 1
                recons.append(model.first_stage_model.decode(
                    latent[:, :, start_frame:end_frame].contiguous(),
                    clear_fake_cp_cache=clear_fake_cp_cache,
                ))
            recon = torch.cat(recons, dim=2).to(torch.float32)
            samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

            motion_image = samples[:, -n_motion_frame:].permute(0, 2, 1, 3, 4).contiguous().to(
                dtype=torch.bfloat16, device="cuda") * 2 - 1
            mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
            mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
            mask_image = model.encode_first_stage(mask_image, None).permute(0, 2, 1, 3, 4).contiguous()
            mask_image = torch.concat(
                [mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1
            )
            video.append(samples[:, n_motion_frame:])

        video = torch.cat(video, dim=1)[:, :length]

        # Writes <work_dir>/000000_with_audio.mp4 (+ removes the no-audio temp).
        save_video_as_grid_and_mp4_with_audio(
            video, work_dir, driving_audio, fps=args.sampling_fps, is_padding=is_padding
        )
        produced = os.path.join(work_dir, "000000_with_audio.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        shutil.copyfile(produced, save_path)
        return save_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate one Hallo3 clip.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--s2-config", required=True)
    p.add_argument("--inference-config", required=True)
    p.add_argument("--source-image", required=True)
    p.add_argument("--driving-audio", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    models = load_models(a.model_dir, a.s2_config, a.inference_config, seed=a.seed)
    generate_video(models, a.source_image, a.driving_audio, a.save_path,
                   prompt=a.prompt, seed=a.seed)


if __name__ == "__main__":
    main()
