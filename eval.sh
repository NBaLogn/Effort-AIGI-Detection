# =============================================
# 3. EVALUATION OF FINE-TUNED MODEL
# =============================================

uv run DeepfakeBench/training/evaluate_finetune.py \
	--detector_config \
		DeepfakeBench/training/config/detector/effort_finetune.yaml \
	--weights \
		DeepfakeBench/training/weights/finetuned/newBatchFaces.pth \
	--test_dataset \
		DeepfakeBench/training/facedata/DiffFace_even_cropped/ADM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/DDIM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/DDPM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/DiffSwap \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/Inpaint \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/LDM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/PNDM \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv15_DS0.3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv15_DS0.5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv15_DS0.7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv21_DS0.3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv21_DS0.5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/SDv21_DS0.7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_1_5_text2img_p3g7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_1_5_text2img_p4g5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_1_5_text2img_p5g3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_2_1_text2img_p0g5 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_2_1_text2img_p1g7 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/stable_diffusion_v_2_1_text2img_p2g3 \
        DeepfakeBench/training/facedata/DiffFace_even_cropped/Wild \
	--output_dir evaluation_results