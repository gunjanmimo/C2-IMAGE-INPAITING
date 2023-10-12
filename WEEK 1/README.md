# WEEK 1: IMAGE INPAINTING WITH LAPLACIAN EQUATION

## THINGS TO UNDERSTAND 

### 1. IMAGE REGIONS

Divide image into two regions: 

**Region A**: The known region where no inpainting is needed. 

**Region B**: The region to be inpainted.
![ALT TEXT](./src/region%20A%20B.png)


### 2. BOUNDARY CONDITIONS 

For pixels on the boundary of the image, not all four neighbors exist. To handle this, you can introduce ghost cells or pad the boundary.

By adding a padding (or ghost boundary) around the image, we are essentially creating an extended version of our domain. This extended domain assists in applying the boundary conditions. Imagine this ghost boundary as a single pixel-wide frame around the image.

**North Side (Top boundary)**:
For the topmost row of the image (excluding the ghost boundary):

$U (i,j) −U (i−1,j)=0$

This implies that the difference in pixel value between a pixel and its northern neighbor (in the ghost boundary) is zero, suggesting a zero gradient or Neumann boundary condition. In other words, the pixel value doesn't change as you move northwards at the top boundary.


​**South Side (Bottom boundary)**:
For every pixel on the bottom boundary:

$U(ni+1,j)​−U(ni,j)​=0$

Here, $U(ni+1,j​)$ is the value at the bottom ghost boundary, and $U(ni,j​)$ is the value of the last pixel row of the original image.

similarly, 

**West Side (Left boundary)**:

$U(i,1)​=U(j,2​)$

**East Side (Right boundary)**:

$U(i,nj+1​)=U(i,nj​)$


### 3. PROCESS OF FILLING INNER PIXEL VALUES 

For the inner pixels (those not on the boundary), the Laplace equation is discretized using a standard finite difference approach. This approach creates approximations for the partial derivatives.

**Mathematics of the Laplace Equation Discretization**:

The 2D Laplace equation is given by:

$∇^2u=0$

Where $∇^2$ is the Laplacian operator, in 2D it's expressed as:

$∇^2u= ∂^2u/∂x^2+∂^2u/∂y^2$

​

Using finite differences, the second order partial derivatives can be approximated as:

For $∂^2u/∂x^2$ at point $(i,j)$:

$∂^2u/∂x^2  \approx \frac{U(i+1,j)​−2U(i,j)​+U(i−1,j)​}{h_i^2} $

For $∂^2u/∂y^2$ at point $(i,j)$:

$∂^2u/∂x^2  \approx \frac{U(i,j+1)​−2U(i,j)​+U(i,j-1)​}{h_i^2} $


Assuming a uniform grid where $h_i=h_j=h$

$\frac{U(i+1,j))​+U(i−1,j)+U(i,j+1)+U(i,j-1)​−4U(i,j}{h_i^2}=0$

After rearranging, we get the standard five-point stencil for the Laplace equation:

$​4U(i,j)-U(i+1,j)-U(i−1,j)-U(i,j+1)-U(i,j-1)=0$


Where:

$U(i,j)$ is the value at the current point.

$U(i+1,j)$ is the value of the point to the right.

$U(i−1,j$ is the value of the point to the left.

$U(i,j+1)$ is the value of the point above.

$U(i,j-1)$ is the value of the point below.
### 4. Next step



After defining the boundary conditions and handling the inner pixels using the five-point stencil for the Laplace equation, the next major step is the assembly of the system of equations in the form $A \times u = b$, and then solving it. 

### Forming the Matrix \(A\):

1. **Assembly**: 
The system is assembled in a matrix form. The matrix $A$ is sparse since each pixel is mainly influenced by its immediate neighbors (North, South, East, West, and itself). 

    - The `idx_Ai` and `idx_Aj` lists contain the row and column indices of the non-zero values in the matrix \(A\), respectively. 
    - The list `a_ij` contains the non-zero values themselves.
  
    For inner pixels where inpainting is required:
    - The diagonal (corresponding to the pixel itself) has a value of 4.
    - The immediate neighbors (North, South, East, West) have a value of -1. 
   
    For inner pixels outside the inpainting region:
    - The diagonal has a value of 1, and the right-hand side $b$ will have the original pixel value from the image.

2. **Sparse Matrix Creation**: 
The matrix \(A\) is large but contains many zeros. To save memory and computational effort, a sparse representation is used. The function `sparse` is used to convert the index-value pairs (`idx_Ai`, `idx_Aj`, and `a_ij`) into a sparse matrix \(A\).

### Forming the Vector \(b\):

The vector $b$ is populated as you traverse the image:

- For boundary pixels, it's set to 0 
  
- For inner pixels in the inpainting region (Region B), it's also set to 0 since we're solving the homogeneous Laplace equation in this region.

- For inner pixels outside of the inpainting region (Region A), it's set to the pixel's value from the original image.

### Solving the Linear Equation:

- **Sparse Solver**: The system $A \times u = b$ is then solved using a sparse linear system solver (`spsolve`). Sparse solvers are efficient for these kinds of problems where the matrix $A$ has a lot of zeros.

- **Reshaping the Solution**: The solution $x$ is a vector. It's reshaped back into an image (matrix form) using the `reshape` function. 

### Extracting the Inpainted Image:

Finally, the ghost boundaries are removed, and the central portion of the resulting matrix represents the inpainted image, which is then returned.

### Mathematical Summary:

For the Laplace equation in two dimensions, the five-point stencil discretizes the equation as:

$ u_{i,j} = \frac{1}{4} (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}) $

This results in the diagonal value of 4 and the off-diagonal values of -1. When this is applied to all pixels, it forms the system $A \times u = b$, which is then solved to get the inpainted image $u$.


# Setup
Clone repo and `cd` into the project folder:

 ```
git clone https://github.com/gunjanmimo/C2-IMAGE-INPAITING.git
cd C2-IMAGE-INPAITING
```

Create and activate virtual environment:

```
python3 -m venv <virtual_environment_name>
source <virtual_environment_name>/bin/activate
```

Install packages from `requirements.txt`:

```
pip install --upgrade pip wheel
pip install -r requirements.txt
```
## Repository contents

```
.
└── C2-IMAGE-INPAITING/
    └── WEEK 1/
        ├── sol_Laplace_Equation_Axb.py # main inpainting functions
        ├── start.py                    # test codes
        ├── start.ipynb                 # notebook with visualization
        ├── README.md                   # our approach and setup details 
        ├── IMAGES/                     # image directory
        │   ├── image{N}_toRestore.jpg
        │   ├── image{N}_mask.jpg
        │   ├── example {N}.jpg
        │   └── example {N}_mask.jpg
        └── src/                        # images for README file
            └── images for readme

```