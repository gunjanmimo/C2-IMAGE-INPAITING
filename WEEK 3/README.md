# WEEK 3: IMAGE SEGMENTATION USING CHAN VESE METHOD 

Before diving deep into Chan-Vese segmentation method, you will discuss about **Mumford-Shah** segmentation problem to understand the background of the problem. 

The Chan-Vese method is inspired by the Mumford-Shah model. They approximate the image by:

They suggest selecting this edge set C as the segmentation boundary. Mumford-Shah implies 4 criteria:
1) The segmentation results in an image: $f \rightarrow u$
2) The segmented image is like the original image: $\int_\Omega (f(x)- u(x))^2 \, dx$
3) The segmented regions are homogeneous: $\int_{\frac{\Omega}{C}}\left|\nabla u(x)\right|^2dx$
4) The segmented regions have smooth boundaries: $\arg_{u,C}(\mu \text{Length}(C))$

Summing the criteria, we get 3 terms:
$\arg_{u,C}(\mu \text{Length}(C)) + \lambda \int_\Omega (f(x)- u(x))^2 \, dx + \int_{\frac{\Omega}{C}}\left|\nabla u(x)\right|^2dx$

Where C is the edge set curve and u is discontinuous.

1: Ensures regularity of C.
2: Ensures u to be close to f.
3: Ensured u is differentiable on $\frac{\Omega}{C}$.

Mumford-Shah models are complicated and computationally expensive. As a simplification, we consider a piecewise constant formulation and a fifth criterion:
5) The segmentation image has two regions:
$\arg_{u,C}(\mu \text{Length}(C)) + \lambda \int_\Omega (f(x)- u(x))^2 \, dx + \int_{\frac{\Omega}{C}}\left|\nabla u(x)\right|^2dx$

Where u is allowed to have only 2 values:
- u(x) = u₁(x) if x is in C₁
- u(x) = u₂(x) if x is not in C₂



So we can rewrite the equation as:
$\arg_{u,C}(\mu \text{Length}(C)) + v \text{Area (inside}(C)) + \lambda_1 \int_{\text{inside}(C)} (f(x) - c_1)^2 \, dx + \lambda_2 \int_{\text{outside}(C)} (f(x) - c_2)^2 \, dx$

By finding a local minimizer of this problem, we obtain a segmentation as the best two-phase piecewise approximation u of the image f.

To minimize the function, we need to minimize over all set boundaries C:

C = $\{x \in \Omega : \varphi(x) = 0 \}$

The inside and outside of C are distinguished by $\varphi$ which is a level set function for a circle of radius r:

$\varphi(x) = r - \sqrt{x_1^2 + x_2^2}$

For a given C, there is more than one possible level set representation. If $\varphi$ is a level set of $\Omega$, then so is any other function $\psi$ having the same sign:

$\text{sign}(\psi(x)) = \text{sign}(\psi(x))$

=> $\arg_{c_1,c_2,\varphi}\mu \int_{\Omega}  \delta \left(\varphi(x)|\nabla\varphi(x)|\right) \, dx +  v \int_{\Omega} H \left(\varphi(x)\right) \, dx +  \lambda_1 \int_{\Omega}  \left|f(x) - c_1\right|^2 H( \varphi(x)) \, dx + \lambda_2 \int_{\Omega}  \left|f(x) - c_2\right|^2 (1 - H( \varphi(x))) \, dx$

Where H denotes the Heaviside function and the $\delta$ the Dirac mass, its distributional derivative,

$H(t) = \begin{cases}
   1 & \text{if } t \geq 0 \\
   0 & \text{if } t < 0
\end{cases}$

$\delta(t) = \frac{d}{dt} H(t)$

Length of C is obtained as the total variation of $H(\varphi)$.

$\text{Length(C)} =  \int_{\Omega} |\nabla H \varphi(x)| \, dx =  \int_{\Omega}\delta( \varphi(x)) |\nabla \varphi(x)| \, dx.$

The minimization is solved by alternatingly updating $c_1$, $c_2$, and $\varphi$. For a fixed $\varphi$, the optimal values of $c_1$ and $c_2$ are the region averages:

$c_1 = \frac{\int_{\Omega} f(x) H(\varphi(x)) \, dx}{\int_{\Omega} H(\varphi(x)) \, dx}$
$c_2 = \frac{\int_{\Omega} f(x) (1 - H(\varphi(x))) \, dx}{\int_{\Omega}( 1 -  H(\varphi(x))) \, dx}$

For the minimization with respect to $\varphi$, H is regularized as:

$H_\epsilon(t) = \frac{1}{2} \left(1 + \frac{2}{\pi} \arctan\left(\frac{t}{\epsilon}\right)\right)$






