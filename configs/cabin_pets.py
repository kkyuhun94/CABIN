dataset = "pets"
model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
logdir = f"{dataset}-baselines/cabin"
prompt_templates = "prompt_generation/prompt_templates"

EXP_NAME_1 = "cabin-phase1"
EXP_NAME_2 = "cabin-phase2"

synthetic_dir_1 = "aug/cabin-phase1/{dataset}-{seed}-{examples_per_class}"
synthetic_dir_2 = "aug/cabin-phase2/{dataset}-{seed}-{examples_per_class}"
num_epochs_1 = 50
num_epochs_2 = 50
synthetic_probability_1 = 1.0
synthetic_probability_2 = 0.5
num_synthetic_1 = 80
num_synthetic_2 = 30
phase_name_1 = "Phase_div"
phase_name_2 = "Phase_key"

guidance_scale = 7.5
num_inference_steps = 30
aug = 'cabin'
examples_per_class = [1]

num_trials = 3
lr = 0.00001
image_size = 256
classifier_backbone = "resnet50" # options: resnet50, vit, vit_l
iterations_per_epoch = 200
batch_size = 32
num_workers = 10
device = 'cuda'
device_num = 0