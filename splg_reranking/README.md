# Reranking using SuperPoint and LightGlue

Superpoint finds local features and LightGlue matches local features between images.
An homography is estimated (ransac) using the mathed features, then the number 
of inliers is used as a score for reranking. This does work very well and it's robust.