We often need to transmit and store the images in many applications. Smaller the image, less is the cost associated with transmission and storage. 
So we often need to apply data compression techniques to reduce the storage space consumed by the image.
Two approaches for this are to apply Singular Value Decomposition (SVD) and Eigen Value Decomposition(EVD) on the image matrix.In these methods, digital image is given to SVD/EVD. 
They refactors the given digital image into three matrices. Singular values or eigen values are used to refactor the image and at the end of this process, image is represented with smaller set of values, hence reducing the storage space required by the image. Let us deep dive in to these methods and compare them.




