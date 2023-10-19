# WEEK 2 | POISSON EDITING 

Poisson editing is a method often used in image manipulation and computer graphics to seamlessly blend one region of an image into another. The technique is based on solving the Poisson equation, which in the context of image editing, allows us to preserve the gradients (or "details") of a source region while adapting to the large-scale structures of a target region.


### NOTATIONS 
$S$  = Source Image, the image or region from which you want to copy details. 

$T$ = Target image, the image where you want to paste or blend the details from the source.

$M$ = Mask, a binary mask is used to define the region within the source image that you want to copy. Pixels with value 1 inside the mask represent the region of interest from the source, while pixels with value 0 are ignored.

$V$ = Blended or Resulting image, the final image obtained after the editing process.


### GRADIENT OF IMAGE AND LAPLACIAN EQUATION

**Gradient**: The gradient of an image provides information about the direction and magnitude of the highest rate of change in intensity at each pixel location.

In a grayscale image (ignoring color for simplicity), the gradient at each pixel (x,y) can be represented as a 2D vector:

$\nabla I(x,y) = [\frac{\partial I}{\partial x},\frac{\partial I}{\partial y} ] $

In discrete space, the partial derivatives can be approximated using finite differences:

$\frac{\partial I}{\partial x} \approx I(x+1,y) - I(x,y) $ 

$\frac{\partial I}{\partial y} \approx I(x,y+1) - I(x,y) $ 

**Laplacian**: The Laplacian is a second-order differential operator that measures the difference between the value of a function at a point and the average of its neighbors. In the context of images, it gives a measure of how much a pixel stands out from its neighbors.

The Laplacian of an image $I$ is defined as the divergence of its gradient:

$\Delta I = \nabla . \nabla I$

So, Laplacian can be written as 

$\Delta I =  [\frac{\partial^2 I}{\partial x^2},\frac{\partial^2 I}{\partial y^2} ]$

so, we can write the change of this first derivative as

$\frac{\partial^2 I}{\partial x^2} \approx (I(x+1,y) - I(x,y) )- (I(x,y) - I(x-1,y)) $

$\frac{\partial^2 I}{\partial y^2}  \approx (I(x,y+1) - I(x,y) )- (I(x,y) - I(x,y-1)) $

Simplifying this:

$\Delta I(x,y) = I(x+1,y)+ I(x-1,y) + I(x,y+1)+I(x,y-1) - 4I(x,y)$


### Role of the Laplacian in Poisson Image Editing:

The main idea behind Poisson Image Editing is to paste the details (or the internal structure) of a source image region onto a target image in a way that is seamless and consistent with the boundary of the target region.

In essence, instead of copying the pixel values directly from the source to the target, we aim to transfer the **features** or **details** represented by the Laplacian.

The Laplacian of an image gives a measure of the local variation at each pixel with respect to its neighbors. In other words, it captures the **details** or **features** of the image. 

The Laplacian ensures that we do so while allowing the overall intensity to adjust and blend with the surrounding target region.

### Role of the Poisson Equation in Poisson Image Editing:

In Poisson Image Editing, the Poisson Equation is written as:

$\Delta V = \Delta S $

here $\Delta$ denotes the Laplacian. $\Delta V$ represents the Laplacian of the resulting image, and $\Delta S $ is the Laplacian of the source region.

The equation ensures that the local features or details (as captured by the Laplacian) of the source $S$ are transferred to the resulting image $V$. 


### BOUNDARY CONDITIONS
Boundary conditions are crucial in Poisson Image Editing to ensure that the blend between the source and the target images appears seamless. In the context of the Poisson Image Editing problem, the boundary conditions are related to the region of the image that's being edited. 

In Poisson Image Editing, the region where the source image is to be blended into the target image is usually defined by a mask and is commonly denoted as $Ω$. The boundary of this region is represented as $∂Ω$.

The boundary conditions are imposed to ensure that the edited region seamlessly integrates with the surrounding area in the target image. Without this condition, you might observe artifacts or noticeable seams where the source and target images meet.

The boundary condition can be mathematically represented as:

$H(x,y)=A(x,y) \ for \ all (x,y)∈∂Ω$

Here:

$H$ represents the final blended image.

$A$ is the target image.

$(x,y)$ are the coordinates of a pixel.

The equation essentially says that, on the boundary $∂Ω$, the resulting image 
$H$ should exactly match the target image $A$.



# STEP BY STEP PSEUDO CODE
