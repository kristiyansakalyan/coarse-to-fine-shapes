- PVD uses PVCNN as backbone
- PVCNN is built on top of PointNet

# TODO LIST:
- Read PointNet paper
- Read PVCNN paper
- Understand PVD and the underlying model


- Read AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE to understand patch by patch manner and encoding and everything
- Implement patch by patch manner
- Overfit on one sample
- Visualize results

# Remarks:
- Cannot reproduce the results from the paper - Choices for beta_start, beta_end and schedule strategy for car do not produce good results.
- Scale matters for the visualization!! Since we downsample to 2048, we need smaller scale.