Put the trained YOLO26 depth/detection/road joint `best_joint.pt` and matching YOLO26 `yolo26s.pt` detector template here.
No weights are distributed in this repository. Override with `CHECKPOINT` and `MODEL_TEMPLATE` in `run_pseudo.sh`.
A plain COCO detector or a DAv2-MTL checkpoint is incompatible with this loader.
The class order must be Golfcart, Person, Tree, Undef_obj. RGB and corresponding NIR images are required by the existing fusion models.
