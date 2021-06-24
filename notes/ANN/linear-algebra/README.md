# Table of Contents

1. [Notations](#notations)
   1. [Norms](#norms)
   2. [Diagonal, Symmetric and orthogonal matrices](#dso)
2. [System of linear equations](#linear-equations-system)
   1. [Gaussian elimination](#gauss-elimination)
   2. [Determinant of a matrix](#determinant)
      1. [Physical meaning of determinant](#physical-meaning-of-determinant)
   3. [Finding Inverse](#inverse)
      1. [Singular Matrix](#singular)
   4. [Linear dependence and Span](#span)
   5. [EigenDecomposition](#eigendecomposition)
      1. [finding eigenvalues and eigenvectors through characteristic equation](#characteristic-equation)
   6. [Properties of Eigendecomposition](#eig-prop)
   7. [Singular Value Decomposition](#svd)
   8. [The Moore-Penrose Pseudoinverse](#pseudoinverse)
   9. [Trace Operator](#trace)



# Notations<a name="notations"></a>

- matrices - uppercase bold math fonts, vectors - lowercase bold math fonts.
- addition of matrix and vector
  - $C_{ij} = A_{ij} + b_j$, $\pmb{C} = \begin{bmatrix} A_{11}+b_1 & A_{12}+b_2 & \cdots & A_{1n}+b_n \\ A_{21}+b_1 & A_{22}+b_2 & \cdots & A_{2n}+b_n \\ & \vdots \\ A_{m1}+b_1 & A_{m2}+b_2 & \cdots & A_{mn}+b_n  \end{bmatrix} = \pmb{A} +\begin{bmatrix} --\pmb{b}-- \\ --\pmb{b}-- \\ \vdots  \\ --\pmb{b}-- \end{bmatrix}_{m \times n}  $
  - this copying of $\pmb{b}$ is called as **broadcasting** the vector.
- matrix product is also known as **element-wise** product or **Hadamard** product. $\pmb{C} = \pmb{A}\odot \pmb{B}$



## Norms<a name="norms"></a>

- the $L^p$ norm : $||\pmb{x}||_p = \left(\sum\limits_{i} |x_i|^p \right)^{\frac{1}{p}}$
- the $L^2$  (pronounced as L-two-norm) is called the **euclidean norm**, or magnitude of a vector. the **squared** euclidean norm is obtained from $\pmb{x}^T\cdot \pmb{x}$
- Sometimes, the size of the vector is measured by counting its number of nonzero elements. 
  - this is **informally** referred to as **L$^0$ norm**.
- a **max norm** is called the **L**$^{\infty}$ = $\textrm{max}_i|x_i|$
- the size of a **matrix** can also be measured by something called as a **frobenius norm**. $||A_F|| = \sqrt{\sum\limits_{i,j} A^2_{i,j} }$
  - this is analogous to the $L^2$  norm of a vector
- The **dot product of two vectors** can be rewritten **in terms of norms** : $\pmb{x}^T \pmb{y} = ||\pmb{x}||_2 ||\pmb{y}||_2 cos\left(\theta \right)$



## Diagonal, Symmetric and orthogonal matrices<a name="dso"></a>

- diagonal matrices - **if and only if** $D_{i,j} = 0 \, \forall \, i,j \,\,, i \ne j$
- diag($\pmb{v}$) = diagonal matrix with diagonal elements = elements of the vector $\pmb{v}$ , or $D_{i,i} = v_i \,,\, D_{i,j} = 0 \, \forall \, i,j \,\,, i \ne j$
- It is possible to construct a rectangular diagonal matrix, though non-square diagonal matrices do not have inverses.
- symmetric matrix , $\pmb{A} = \pmb{A}^T$ can only happen if m = n , in $\pmb{A}_{m \times n}$ , which means that $A_{i,j} = A{j, i}\,\, \forall \, i,j,\,\, \, i \ne j $
- 2 vectors $\pmb{x}$  and $\pmb{y}$  are orthogonal to each other if $\pmb{x}^T \pmb{y} = 0 $, i.e. their dot product is 0.
- a **square matrix** is said to be ***orthogonal*** if: ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%5ETA%7D%20%3D%20%5Cpmb%7BAA%5ET%7D%20%3D%20%5Cpmb%7BI%7D_%7Bn%20%5Ctimes%20n%7D)
  - this implies that ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%5E%7B-1%7D%7D%20%3D%20%5Cpmb%7BA%5ET%7D)
  - observe that for such a matrix,  all **rows** are **orthonormal** and **columns** are **orthonormal**.
    - **orthonormality** is for vectors, and it means that all vectors have $L^2$ norm = 1 and are pairwise orthogonal.
    - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?A%20%3D%20%5Cbegin%7Bbmatrix%7D%20--R_1--%20%5C%5C%20--R_2--%20%5C%5C%20%5Cvdots%20%5C%5C%20--R_m--%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%20%5C%2C%20A%5ET%20%3D%20%5Cbegin%7Bbmatrix%7D%20%7C%20%26%20%7C%20%26%20%26%20%7C%20%5C%5C%20%7C%26%20%7C%20%26%20%26%20%7C%20%5C%5C%20R_1%26%20R_2%20%26%20%5Ccdots%20%26%20R_m%20%5C%5C%20%7C%26%20%7C%20%26%20%26%20%7C%20%5C%5C%20%7C%26%20%7C%26%20%26%20%7C%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20A%5Ccdot%20A%5ET%20%3D%20%5Cbegin%7Bbmatrix%7D%20R_1%5E2%20%26%20R_1R_2%20%26%20%5Ccdots%20%26%20R_1R_m%20%5C%5C%20R_1R_2%20%26%20R_2%5E2%20%26%20%5Ccdots%20%26%20R_2R_m%20%5C%5C%20%5Cvdots%20%5C%5C%20R_mR_1%20%26%20R_mR_2%20%26%20%5Ccdots%20%26%20R_m%5E2%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20%5Ctextrm%7B%20now%20if%20this%20is%20equal%20to%20I%2C%20%7D%20R_i%5E2%20%3D%201%20%28%5Ctextrm%7Bmeaning%20each%20row%20is%20of%20norm%20%3D%201%7D%29%20%5C%2C%2C%5C%2C%20R_iR_j%20%3D%200%20%28%5Ctextrm%7B%20any%202%20rows%20are%20orthogonal%7D%29)
    - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%7C%20%26%20%7C%20%26%20%5Ccdots%20%26%20%7C%20%5C%5C%20%7C%20%26%20%7C%20%26%20%5Ccdots%20%26%20%7C%20%5C%5C%20C_1%20%26%20C_2%20%26%20%5Ccdots%20%26%20C_n%20%5C%5C%20%7C%20%26%20%7C%20%26%20%5Ccdots%20%26%20%7C%20%5C%5C%20%7C%20%26%20%7C%20%26%20%5Ccdots%20%26%20%7C%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7BA%5ET%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20--C_1--%20%5C%5C%20--C_2--%20%5C%5C%20%5Cvdots%20%5C%5C%20--C_n--%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20%5Cpmb%7BA%5ET%7D%20%5Codot%20%5Cpmb%7BA%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20C_1%5E2%20%26%20C_1C_2%20%26%20%5Ccdots%20%26%20C_1C_n%20%5C%5C%20C_2C_1%20%26%20C_2%5E2%20%26%20%5Ccdots%20%26%20C_2C_n%20%5C%5C%20%5Cvdots%20%5C%5C%20C_nC_1%20%26%20C_nC_2%20%26%20%5Ccdots%20%26%20C_n%5E2%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%5Ccdots%20%26%200%20%5C%5C%200%20%26%201%20%26%200%20%5Ccdots%20%26%200%20%5C%5C%20%5Cvdots%20%5C%5C%200%20%26%200%20%26%200%20%5Ccdots%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20C_i%5E2%20%3D%201%20%5C%2C%2C%5C%2C%20C_iC_j%20%3D%200)
    - 
- 



# System of linear equations<a name="linear-equations-system"></a>

- $\pmb{Ax} = \pmb{b}$, where $\pmb{A}_{m\times n}$ and $\pmb{b}_{m \times 1}$ are known and $\pmb{x}$ is to be found out.
- $\begin{bmatrix} A_{11} & A_{12} & \cdots & A_{1n} \\ A_{21} & A_{22} & \cdots & A_{2n} \\ & \vdots \\ A_{m1} & A_{m2} & \cdots & A_{mn}  \end{bmatrix}\cdot \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} =  \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$
  $\forall j, j \in [1,2,\cdots m] \,, \, \sum\limits_{i=1}^n A_{ji}x_i = b_i$
- therefore, we have n variables with m constraints.



## Gaussian elimination<a name="gauss-elimination"></a>

- 
- https://math.ryerson.ca/~danziger/professor/MTH141/Handouts/Slides/gauss.pdf



## Determinant of a matrix<a name="determinant"></a>

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%5C%5C%20c%20%26%20d%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%5Cbegin%7Bvmatrix%7D%20a%20%26%20b%20%5C%5C%20c%20%26%20d%20%5Cend%7Bvmatrix%7D%20%3D%20ad%20-%20bc%20%5Cnewline%20det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%5Cbegin%7Bvmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bvmatrix%7D%20%3D%20a%5Cbegin%7Bvmatrix%7D%20e%20%26%20f%20%5C%5C%20h%20%26%20i%20%5Cend%7Bvmatrix%7D%20-%20b%20%5Cbegin%7Bvmatrix%7D%20d%20%26%20f%20%5C%5C%20g%20%26%20i%20%5Cend%7Bvmatrix%7D%20&plus;%20c%5Cbegin%7Bvmatrix%7D%20d%20%26%20e%20%5C%5C%20g%20%26%20h%20%5Cend%7Bvmatrix%7D%20%3D%20aei%20-%20afh%20-%20b%5Cleft%28di%20-%20fg%20%5Cright%20%29%20&plus;%20cdh%20-%20ceg%20%3D%20aei%20&plus;%20cdh%20&plus;%20bfg%20-%20%28afh%20&plus;%20cdi%20&plus;%20ceg%29)

- as it might be obvious, this is defined only for a square matrix.

- this is the row-wise definition, the column-wise definition is ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%5C%5C%20c%20%26%20d%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%5Cbegin%7Bvmatrix%7D%20a%20%26%20b%20%5C%5C%20c%20%26%20d%20%5Cend%7Bvmatrix%7D%20%3D%20a%28d%29%20-%20c%28b%29%20%5Cnewline%20det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%5Cbegin%7Bvmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bvmatrix%7D%20%3D%20a%5Cbegin%7Bvmatrix%7D%20e%20%26%20f%20%5C%5C%20h%20%26%20i%20%5Cend%7Bvmatrix%7D%20-%20d%20%5Cbegin%7Bvmatrix%7D%20b%20%26%20c%20%5C%5C%20h%20%26%20i%20%5Cend%7Bvmatrix%7D%20&plus;%20g%5Cbegin%7Bvmatrix%7D%20b%20%26%20c%20%5C%5C%20e%20%26%20f%20%5Cend%7Bvmatrix%7D%20%3D%20aei%20-%20afh%20-%20d%5Cleft%28bi%20-%20ch%20%5Cright%20%29%20&plus;%20gbf%20-%20gec%20%3D%20aei%20&plus;%20cdh%20&plus;%20cfg%20-%20%28afh%20&plus;%20cdi%20&plus;%20ceg%29)

- it is then quite obvious that if **a row is all 0's or a column is all 0's** , **determinant is 0**.

- ### Physical meaning of determinant<a name="physical-meaning-of-determinant"></a>

  - for a 2-D vector, a 2 x 2 matrix multiplication transforms the vectors such that the area between the vectors is scaled by a factor equal to the determinant of this square matrix.
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Bv_1%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%203%20%5C%5C%204%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7BA%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%26%201%20%5C%5C%201%20%26%202%20%5Cend%7Bbmatrix%7D%20%5Ctextrm%7B%2C%20area%7D%28%5Cpmb%7Bv_1%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%7D%29%20%3D%20%7C%7C%5Cpmb%7Bv_1%7D%7C%7C%20%5Ccdot%20%7C%7C%5Cpmb%7Bv_2%7D%20%7C%7C%20sin%28%5Ctheta%29%20%5Cnewline%20sin%28%5Ctheta%29%20%3D%20%5Csqrt%7B1%20-%20cos%5E2%28%5Ctheta%29%7D%20%3D%20%5Csqrt%7B1%20-%20%5Cleft%28%20%5Cfrac%7B10%7D%7B5%20%5Csqrt%7B5%7D%20%7D%20%5Cright%20%29%5E2%20%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B5%7D%7D%20%5Cnewline%20%5Ctextrm%7Barea%7D%28%5Cpmb%7Bv_1%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%7D%29%20%3D%205%20%5C%2C%2C%5C%2C%20det.%28%5Cpmb%7BA%7D%29%20%3D%203%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_1%27%7D%20%3D%20%5Cpmb%7BAv_1%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%205%20%5C%5C%204%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%27%7D%20%3D%20%5Cpmb%7BAv_2%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%2010%20%5C%5C%2011%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20%5Ctextrm%7Barea%7D%28%5Cpmb%7Bv_1%27%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%27%7D%29%20%3D%20%7C%7C%5Cpmb%7Bv_1%27%7D%7C%7C%20%5Ccdot%20%7C%7C%5Cpmb%7Bv_2%27%7D%20%7C%7C%20sin%28%5Ctheta%20%27%29%20%3D%20%5Csqrt%7B41%7D%20%5Ccdot%20%5Csqrt%7B221%7D%20%5Ccdot%20%5Csqrt%7B1%20-%20%5Cleft%28%20%5Cfrac%7B94%7D%7B%5Csqrt%7B41%7D%20%5Ccdot%20%5Csqrt%7B221%7D%7D%20%5Cright%29%5E2%20%7D%20%3D%20%5Csqrt%7B%20221%5Ctimes%2041%20-%2094%5E2%7D%20%3D%20%5Csqrt%7B9061%20-%208836%7D%20%3D%20%5Csqrt%7B225%7D%20%3D%2015%20%3D%203%20%5Ctimes%205%20%3D%20det.%28%5Cpmb%7BA%7D%29%20%5Ctimes%20%5Ctextrm%7B%2C%20area%7D%28%5Cpmb%7Bv_1%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%7D%29%20%5Cnewline)
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctherefore%20%7B%5Ccolor%7BDarkRed%7D%20%5C%2C%2C%5C%2C%20%5Ctextrm%7B%2C%20area%7D%28%5Cpmb%7BAv_1%7D%5C%2C%2C%5C%2C%20%5Cpmb%7BAv_2%7D%29%20%3D%20%5Ctextrm%7Barea%7D%28%5Cpmb%7Bv_1%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%7D%29%20%5Ccdot%20det%28%5Cpmb%7BA%7D%29%7D)
  -  for an n-D vector, an  *n x n* matrix multiplication transforms the vectors such that the volume of the space in-between the vectors is scaled by a factor equal to the determinant of this square matrix. <font color="Red">remaining!!</font>
  - for a visual reference, watch [this video](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=7).
  - hence a 0-determinant transformation matrix will squish the space formed between vectors, such that the volume enclosed = 0.
    - for a 2-D case, the vectors become parallel, 
    - for a 3-D case, the vectors become co-planar.

- 



## Finding Inverse<a name="inverse"></a>

- for a **square matrix**, i.e. m = n, the best way is to find the inverse of the matrix ![equation](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D), i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D%5E%7B-1%7D), such that solution for ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Bx%7D%20%5Ctextrm%7B%20in%20%7D%20%5Cpmb%7BAx%7D%20%3D%20%5Cpmb%7Bb%7D) is ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Bx%7D%20%3D%20%5Cpmb%7BA%5E%7B-1%7Db%7D)

- this method uses the **adjoint matrix** of a given square matrix

  - first we have to define a matrix called **cofactor matrix**
    ![{\displaystyle {\text{cof}}{\begin{bmatrix}a&b&c\\d&e&f\\g&h&i\end{bmatrix}}={\begin{bmatrix}{\begin{vmatrix}e&f\\h&i\end{vmatrix}}&-{\begin{vmatrix}d&f\\g&i\end{vmatrix}}&{\begin{vmatrix}d&e\\g&h\end{vmatrix}}\\-{\begin{vmatrix}b&c\\h&i\end{vmatrix}}&{\begin{vmatrix}a&c\\g&i\end{vmatrix}}&-{\begin{vmatrix}a&b\\g&h\end{vmatrix}}\\{\begin{vmatrix}b&c\\e&f\end{vmatrix}}&-{\begin{vmatrix}a&c\\d&f\end{vmatrix}}&{\begin{vmatrix}a&b\\d&e\end{vmatrix}}\end{bmatrix}}}](https://wikimedia.org/api/rest_v1/media/math/render/png/7a16f37edf5228733b1b9115e67520ef1e42c366)
  - this can also be said in the following other way: ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%20for%20%7D%20%5Cpmb%7BC%7D%20%3D%20cofactor%28%5Cpmb%7BA%7D%29%2C%20%5Ctextrm%7B%20generic%20element%20%7D%20%3D%20C_%7Bij%7D%2C%20%5Ctextrm%7B%20such%20that%20%7D%20det.%28%5Cpmb%7BA%7D%29%20%3D%20%5Csum%5Climits_%7Bj%3D1%7D%5En%20A_%7Bij%7D.C_%7Bij%7D), i.e. multiply the same row or same column to get the determinant.
  - the **adjoint matrix** is defined as the **transpose** of the cofactor matrix, i.e. ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Badj.%28A%29%7D%20%3D%20cofactor%28%5Cpmb%7BA%7D%29%5ET) , hence the product of the nth row of A and the nth column of the adjoint matrix.

- the following is the theorem used to calculate the inverse

  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%20%5Codot%20adj.%28A%29%7D%20%3D%20det.%28%5Cpmb%7BA%7D%29%20%5Cpmb%7BI%7D)
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%20%5Codot%20adj.%28A%29%7D%20%3D%20det.%28%5Cpmb%7BA%7D%29%20%5Cpmb%7BI%7D%20%5Cnewline%20%5Ctextrm%7BProof%3A%20%7D%20%5Ctextrm%7Blet%20%7D%20%5Cpmb%7BC%7D%20%3D%20%5Cpmb%7BA%7D%20%5Codot%20%5Cunderbrace%7B%5Cpmb%7Badj.%28A%29%7D%7D_%7B%5Cpmb%7BB%7D%7D%20%5CRightarrow%20C_%7Bij%7D%20%3D%20%5Csum%5Climits_%7Bk%3D1%7D%5En%20A_%7Bik%7DB_%7Bkj%7D%20%5Ctextrm%7Bfor%20i%20%3D%20j%7D%2C%20B_%7Bkj%7D%2C%20i.e.%5C%2C%20B_%7Bki%7D%20%5Cnewline%5Ctextrm%7B%20is%20the%20cofactor%20of%20%7D%20A_%7Bik%7D%20%28%5Ctextrm%7Bremember%20%7D%20%5Cpmb%7BB%7D%20%5Ctextrm%7B%20is%20the%20tranpose%20of%20the%20cofactor%20matrix%7D%29%20%5Cnewline%20%5Ctherefore%20%5C%2C%2C%5C%2C%20C_%7Bii%7D%20%3D%20det.%28%5Cpmb%7BA%7D%29%20%5Ctextrm%7B%20and%20%7D%20%5Ctextrm%7B%20for%20%7D%20i%20%5Cne%20j%20%5C%2C%2C%5C%2C%20A_%7Bik%7DB_%7Bkj%7D%20%3D%200%20%5Cnewline%20%5Ctherefore%20%5C%2C%2C%5C%2C%20%5Cpmb%7BC%7D%20%3D%20det.%28%5Cpmb%7BA%7D%29%5Cpmb%7BI%7D%20%5Cnewline)
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%20%5Codot%20adj.%28A%29%7D%20%3D%20det.%28%5Cpmb%7BA%7D%29%20%5Cpmb%7BI%7D%20%5Cnewline%20%5Ctextrm%7B%20%5Ctextbf%7BPre%7D-Multiplying%20with%20%7D%20%5Cpmb%7BA%5E%7B-1%7D%7D%20%5Ctextrm%7B%20on%20both%20sides%20%7D%20%5Cnewline%20%5Cpmb%7Badj.%28A%29%7D%20%3D%20%5Cpmb%7BA%5E%7B-1%7D%7D%5Ctimes%20det.%28%5Cpmb%7BA%7D%29%20%5CRightarrow%20%7B%5Ccolor%7Bblue%7D%20%5Cpmb%7BA%5E%7B-1%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bdet.%28%5Cpmb%7BA%7D%29%7D%5Ctimes%20%5Cpmb%7Badj.%28A%29%7D%20%7D)

- for $\pmb{A}$  to be invertible, such that $\pmb{x} = \pmb{A}^{-1}\pmb{b}$ , the matrix $\pmb{A}$  the matrix must be square, i.e. **m = n** and that all the columns be linearly independent. 

- ### Singular Matrix<a name="singular"></a>

  - A **square matrix** with **linearly dependent columns** is known as **singular**.
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%5C%5C%20c%20%26%20d%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20a%28d%29%20-%20c%28b%29%20%5Cnewline%20det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b-ka%20%5C%5C%20c%20%26%20d-kc%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20a%28d-kc%29%20-%20%28b-ka%29%28c%29%20%3D%20ad%20-akc%20-bc&plus;kac%20%3D%20ad%20-%20bc%20%5Cnewline%20det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20aei%20&plus;%20cdh%20&plus;%20cfg%20-%20%28afh%20&plus;%20cdi%20&plus;%20ceg%29%20%5Cnewline%20det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a-kb%20%26%20b%20%26%20c%20%5C%5C%20d-ke%20%26%20e%20%26%20f%20%5C%5C%20g-kh%20%26%20h%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%28a-kb%29%28ei%20-%20fh%29%20-%20%28d-ke%29%28bi-ch%29%20&plus;%20%28g-kh%29%28bf-ce%29%20%3D%20aei-afh%7B%5Ccolor%7Bred%7D-kbei%7D%7B%5Ccolor%7Bblue%7D&plus;kbfh%7D-dbi&plus;dch%7B%5Ccolor%7Bred%7D&plus;kebi%7D%7B%5Ccolor%7Bgreen%7D-kech%7D%20&plus;%20bhf-gce%7B%5Ccolor%7Bblue%7D-khbf%7D%7B%5Ccolor%7Bgreen%7D&plus;khce%7D%20%3D%20aei-afh-dbi&plus;dch&plus;bhf-gce)
  - hence, a single column/row transformation does not change the determinant.
  - however, a row/column swap will negate the determinant, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%5Crightarrow%20%5Ctextrm%7BRow%20shifting%7D%20%5Crightarrow%20det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20b%20%26%20a%20%26%20c%20%5C%5C%20e%20%26%20d%20%26%20f%20%5C%5C%20h%20%26%20g%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20bdi-bfg-a%28ei-fh%29&plus;c%28eg-dh%29%20%3D%20bdi&plus;afh&plus;ceg%20-%20%28bgf&plus;aei&plus;cdh%29%20%3D%20-%28bgf&plus;aei&plus;cdh-%28bdi&plus;afh&plus;ceg%29%20%29%20%3D%20-det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20b%20%26%20c%20%5C%5C%20d%20%26%20e%20%26%20f%20%5C%5C%20g%20%26%20h%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29)
  - hence, according to the definition of a singular matrix, the **determinant of a singular matrix is 0**
    - the linearly dependent column , using column swapping, can be made to be the first column(from the left) , this will negate the determinant of the original singular matrix.
    - this column can now be transformed using column-wise arithmetic transformation, such that it becomes an n-dimensional null vector. but this makes the **determinant = 0**.
    - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20a%20%26%20ka%20%26%20c%20%5C%5C%20d%20%26%20kd%20%26%20f%20%5C%5C%20g%20%26%20kg%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20-det%5Cleft%28%5Cbegin%7Bbmatrix%7D%20ka%20%26%20a%20%26%20c%20%5C%5C%20kd%20%26%20d%20%26%20f%20%5C%5C%20kg%20%26%20g%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20C_1%20%5Crightarrow%20C_1%20-%20kC_2%20%5Cnewline%20%3D%20-det%5Cleft%28%5Cbegin%7Bbmatrix%7D%200%20%26%20a%20%26%20c%20%5C%5C%200%20%26%20d%20%26%20f%20%5C%5C%200%20%26%20g%20%26%20i%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20-0%20%3D%200)

- 





## Linear dependence and Span<a name="span"></a>

- Think of the columns of $\pmb{A}$ as specifying different directions we can travel in from the origin.
- In this view, each element of $\pmb{x}$ specifies how far we should travel in each of these directions, with $x_i$ specifying how far to move in the direction of column i
  - $\pmb{Ax} = \sum\limits_{i=1}^n x_iA_{:, i}$ , since $\pmb{A}$  has n columns, where i'th is denoted by $A_{:, i}$.
- The **span** of a set of vectors is the set of all points obtainable by linear combination of the original vectors.
  - doesn't matter whether the vectors **are orthogonal or not**.
- Determining whether $\pmb{Ax} = \pmb{b}$ has a solution depends on whether $\pmb{b}$ is in the **span of the columns** of $\pmb{A}$. 
  - This particular span is known as the column space, or the range, of $\pmb{A}$.
- for  $\pmb{Ax} = \pmb{b}$ to have a solution, for all values of $\pmb{b} \in \mathbb{R}^{m \times 1} $, the column space of $\pmb{A}$ be all of $\mathbb{R}^{m \times 1}$ .
  - The requirement that the column space of $\pmb{A}$ be all of $\mathbb{R}^{m \times 1}$ implies immediately that $\pmb{A}$ must have at least m columns, that is, n ≥ m.
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BAx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5E%7B%5Cpmb%7Bn%7D%7D%20x_i%5Ccdot%20A_%7B%3A%2Ci%7D%20%5Ctextrm%7B%20this%20will%20have%20n-terms%20%7D%20%5Cnewline%20%5Cmathbb%7BR%7D%5E%7Bm%20%5Ctimes%201%7D%20%5Ctextrm%7B%20has%20dimensionality%20of%20m%2C%20and%20a%20vector-space%20of%7D%20%5Cnewline%20%5Ctextrm%7B%20dimensionality%20m%20can%20be%20engulfed%20only%20in%20a%20%7D%20%5Cnewline%20%5Ctextrm%7Bvector-space%20of%20dimensionality%20k%20if%20%7D%20k%20%5Cge%20m%20%5Cnewline%20%5Ctextbf%7Bhence%2C%20%7D%20%5Cpmb%7Bn%7D%20%5Cge%20%5Cpmb%7Bm%7D)
  - For example, consider a 3 × 2 matrix. 
    - The target $\pmb{b}$ is 3-D, but $\pmb{x}$ is only 2-D, so modifying the value of $\pmb{x}$ at best enables us to trace out a 2-D plane within $\mathbb{R}^3$. 
    - The equation has a solution if and only if  $\pmb{b}$  lies on that plane.
  - Having n ≥ m  is not a sufficient condition, because it is possible for some of the columns to be redundant. 
    - Consider a 2 × 2 matrix where both of the columns are identical.
    - The column space is still just a line and fails to encompass all of $\mathbb{R}^2$ , even though there are two columns.
    - this kind of redundancy is known as **linear dependence**
- the matrix must contain at least one set of m linearly independent columns. 
  - This condition is **both necessary and sufficient** for the system of equations to  have a solution for **every value** of  $\pmb{b}$ .
- No set of m-dimensional vectors can have more than m mutually linearly independent columns, but a matrix with more than m columns may have more than one such set.
- if the opposite occurs, then over-constraining will occur, i.e. number of variables < number of constraints(equations), hence no perfect solution will exist. this commonly happens in machine learning, where m = n(total number of samples) and n = p(total number of features) and more often than not, p < n.





## Eigendecomposition<a name="eigendecomposition"></a>

- Every real symmetric matrix has real-valued eigenvectors and real-valued eigenvalues
  <img src="proofs/real-symmetric-real-eigenvalues.png" />
  
- A **square** matrix is [**singular**](#singular) if and only if **at-least 1 eigenvalue is 0**.

  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?det.%28%5Cpmb%7BA%7D%20-%20%5Clambda%5Cpmb%7BI%7D%29%20%3D%200%20%5Ctextrm%7B%20with%20%7D%20%5Clambda%20%3D%200%20%5Ctextrm%7B%20as%20a%20root%2C%20%7D%20%5CRightarrow%20det.%28%5Cpmb%7BA%7D%29%20%3D%200%20%5CRightarrow%20%5Cpmb%7BA%7D%20%5Ctextrm%7B%20is%20singular.%7D)

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BAv%7D%20%3D%20%5Clambda%5Cpmb%7Bv%7D%20%5CRightarrow%20%28%5Cpmb%7BA%7D%20-%20%5Clambda%5Cpmb%7BI%7D%29%5Cpmb%7Bv%7D%20%3D%200)

  - for the equation ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BAx%7D%20%3D%20%5Cpmb%7B0%7D), if ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D%5E%7B-1%7D) exists, then ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Bx%7D%20%3D%20%5Cpmb%7BA%5E%7B-1%7D0%7D%20%3D%20%5Cpmb%7B0%7D) 
  - since all eigenvectors are non-null vectors, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D%5E%7B-1%7D) **should not exist** , i.e. becomes a singular matrix, hence in the equation ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%28%5Cpmb%7BA%7D%20-%20%5Clambda%5Cpmb%7BI%7D%29%5Cpmb%7Bv%7D%20%3D%200%20%5CRightarrow%20det.%28%5Cpmb%7BA%7D%20-%20%5Clambda%5Cpmb%7BI%7D%29%20%3D%200)
  - this determinant equation is called the **Characteristic equation**

- ### Finding eigenvalues and eigenvectors through characteristic equation<a  name="characteristic-equation"></a>

  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%208%20%26%20-8%20%26%20-2%20%5C%5C%204%20%26%20-3%20%26%20-2%20%5C%5C%203%20%26%20-4%20%26%201%20%5Cend%7Bbmatrix%7D%20%5CRightarrow%20%28%5Cpmb%7BA%7D%20-%20%5Clambda%5Cpmb%7BI%7D%29%5Cpmb%7Bv%7D%20%3D%200%20%5CRightarrow%20det.%28%5Cpmb%7BA%7D%20-%20%5Clambda%5Cpmb%7BI%7D%29%20%3D%200%20%5Cnewline%20det%5Cleft%28%20%5Cbegin%7Bbmatrix%7D%208-%5Clambda%20%26%20-8%20%26%20-2%20%5C%5C%204%20%26%20-3-%5Clambda%20%26%20-2%20%5C%5C%203%20%26%20-4%20%26%201%20-%5Clambda%20%5Cend%7Bbmatrix%7D%20%5Cright%20%29%20%3D%20%288-%5Clambda%29%28%5Clambda%5E2%20&plus;%202%20%5Clambda%20-11%29%20&plus;8%2810-4%5Clambda%29%20-2%283%5Clambda%20-%207%29%20%3D%20-%5Clambda%5E3%20&plus;%206%5Clambda%5E2%20-%2011%20%5Clambda%20-%2088%20&plus;%2080%20&plus;%2014%20%3D%200%20%5CRightarrow%20%5Clambda%5E3%20-%206%5Clambda%5E2%20&plus;%2011%20%5Clambda%20-6%20%3D%200%20%5Cnewline%20%5Clambda%3D1%20%5Ctextrm%7B%20satisfies%20this%20equation%2C%20after%20dividing%20the%20above%20equation%20by%20%7D%20%5Clambda-1%20%5Cnewline%20%5Clambda%5E2%20-5%5Clambda%20&plus;%206%20%3D%200%20%5CRightarrow%20%28%5Clambda%20-2%29%28%5Clambda%20-3%29%20%3D%200%20%5Cnewline%20%7B%5Ccolor%7Bred%7D%20%5Clambda%20%3D%201%2C2%2C3%7D)
  - <img src="proofs/eigenvector-1.png" />
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%205%20%26%20-8%20%26%20-2%20%5C%5C%20-1%20%26%202%20%26%200%20%5C%5C%203%20%26%20-4%20%26%20-1%20%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20v_1%20%5C%5C%20v_2%20%5C%5C%20v_3%20%5Cend%7Bbmatrix%7D%20%3D%200%20%5C%2C%5C%2C%20R_1%20%5Crightarrow%20R_1%20-%202R_3%20%5CRightarrow%20%5Cbegin%7Bbmatrix%7D%20-1%20%26%200%20%26%200%20%5C%5C%20-1%20%26%202%20%26%200%20%5C%5C%203%20%26%20-4%20%26%20-1%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20v_1%20%3D%202v_2%20%5Ctextrm%7B%20and%20%7D%203v_1%20-%204v_2%20-v_3%20%3D%206v_2%20-4v_2%20-%20v_3%20%5CRightarrow%202%20v_2%20%3D%20v_3%20%5Cnewline%20%5Ctherefore%20%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_3%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202v_2%20%5C%5C%20v_2%20%5C%5C%202v_2%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%5C%5C%201%20%5C%5C%202%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_3%7D%5ET.%5Cpmb%7Bv_3%7D%20%3D%204&plus;1&plus;4%20%3D%209%20%5Cnewline%20%5Ctherefore%20%5C%2C%2C%5C%2C%20%5Ctextrm%7B%20the%20corresponding%20unit%20vector%20%7D%20%5C%2C%5C%2C%20%5Cpmb%7Bv_3%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B2%7D%7B3%7D%20%5C%5C%20%5Cfrac%7B1%7D%7B3%7D%20%5C%5C%20%5Cfrac%7B2%7D%7B3%7D%20%5Cend%7Bbmatrix%7D)

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Boptimize%20%7D%20f%28%5Cpmb%7Bx%7D%29%20%3D%20%5Cpmb%7Bx%5ET%7D%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%5C%2C%2C%5C%2C%20%5Ctextrm%7B%20subject%20to%20%7D%20%7C%7C%5Cpmb%7Bx%7D%7C%7C_2%20%3D%201%20%5Cnewline%20%5Ctextrm%7B%20any%20vector%20can%20be%20represented%20as%20a%20linear%20combination%20of%20its%20eigenspace%7D%20%5Cnewline%20%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7Bv_i%7D%20%5CRightarrow%20%5Cpmb%7Bx%7D%5ET%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5E2%20%5C%2C%2C%5C%2C%20%5Ctextrm%7B%20since%20%7D%20%7C%7C%5Cpmb%7Bx%7D%7C%7C_2%20%3D%201%20%5CRightarrow%20%5Cpmb%7Bx%7D%5ET%5Cpmb%7Bx%7D%20%3D%201%20%5CRightarrow%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5E2%20%3D%201%20%5Cnewline%20%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7Bv_i%7D%20%5CRightarrow%20%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7BA%7D%5Cpmb%7Bv_i%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%20%5C%2C%20%5Clambda_i%20%5Cpmb%7Bv_i%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bx%5ET%7D%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5E2%20%5C%2C%20%5Clambda_i%20%5Cnewline%20%5Ctextrm%7B%20for%20an%20expression%20%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20k_i%20%5C%2C%20%5Clambda_i%20%5Ctextrm%7B%20with%20%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20k_i%20%3D%201%20%5Ctextrm%7B%20and%20%7D%20k_i%20%5Cge%200%5C%2C%2C%5C%2C%20%5Ctextrm%7B%20the%20max%20value%20is%20%7D%20%5Ctextrm%7Bmax.%7D%28%5Clambda_i%29%5Cnewline%20%5Ctextrm%7B%20and%20min.%20value%20is%20%7D%20%5Ctextrm%7Bmin.%7D%28%5Clambda_i%29)

- positive semi-definite
  
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%20any%20vector%20can%20be%20represented%20as%20a%20linear%20combination%20of%20its%20eigenspace%7D%20%5Cnewline%20%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7Bv_i%7D%20%5CRightarrow%20%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7BA%7D%5Cpmb%7Bv_i%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%20%5C%2C%20%5Clambda_i%20%5Cpmb%7Bv_i%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bx%5ET%7D%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5E2%20%5C%2C%20%5Clambda_i%20%5Cnewline%20%5Ctextrm%7Bsince%20all%20eigenvalues%20are%20%7D%20%5Cge%200%20%5Ctextrm%7B%20and%20a%20square%20is%20also%20the%20same%2C%20%7D%20%5Cpmb%7Bx%5ET%7D%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%5Cge%200)
  
- positive definite
  
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%20any%20vector%20can%20be%20represented%20as%20a%20linear%20combination%20of%20its%20eigenspace%7D%20%5Cnewline%20%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7Bv_i%7D%20%5CRightarrow%20%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5C%2C%20%5Cpmb%7BA%7D%5Cpmb%7Bv_i%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%20%5C%2C%20%5Clambda_i%20%5Cpmb%7Bv_i%7D%5C%2C%2C%5C%2C%20%5Cpmb%7Bx%5ET%7D%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5E2%20%5C%2C%20%5Clambda_i%20%5Cnewline%20%5Ctextrm%7B%20if%20%7D%20%5Cpmb%7Bx%5ET%7D%5Cpmb%7BA%7D%5Cpmb%7Bx%7D%20%3D%200%5C%2C%2C%5C%2C%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20c_i%5E2%20%5C%2C%20%5Clambda_i%20%3D%200%20%2C%20%5Cbecause%20%5C%2C%2C%5C%2C%20%5Clambda_i%20%3E%200%20%5C%2C%5C%2C%2C%5C%2C%5C%2C%20%5Ctherefore%5C%2C%2C%5C%2C%20%5Cforall%20i%5C%2C%2C%5C%2C%20c_i%20%3D%200%20%5CRightarrow%20%5Cpmb%7Bx%7D%20%3D%20%5Cpmb%7B0%7D)
  
- 



### 	





## Properties of Eigendecomposition<a name="eig-prop"></a>

- a vector that is a linear combination of 2 vectors having the same eigenvalues, will also have the exact eigenvalue.![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Bv_1%7D%20%5Ctextrm%7B%20and%20%7D%20%5Cpmb%7Bv_2%7D%20%5Ctextrm%7B%20have%20%7D%20%5Clambda%20%5Ctextrm%7B%20as%20their%20eigenvalues%7D%20%5CRightarrow%20%5Cpmb%7BAv_1%7D%20%3D%20%5Clambda%20%5Cpmb%7Bv_1%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7BAv_2%7D%20%3D%20%5Clambda%20%5Cpmb%7Bv_2%7D%20%5Cnewline%20%5Cpmb%7Bu%7D%20%3D%20c_1%5Cpmb%7Bv_1%7D%20&plus;%20c_2%5Cpmb%7Bv_2%7D%20%5CRightarrow%20%5Cpmb%7BAu%7D%20%3D%20c_1%5Cpmb%7BAv_1%7D%20&plus;%20c_2%5Cpmb%7BAv_2%7D%20%3D%20c_1%5Clambda%20%5Cpmb%7Bv_1%7D%20&plus;%20c_2%20%5Clambda%20%5Cpmb%7Bv_2%7D%20%3D%20%5Clambda%5Cleft%28c_1%5Cpmb%7Bv_1%7D%20&plus;%20c_2%5Cpmb%7Bv_2%7D%20%5Cright%20%29%20%3D%20%5Clambda%20%5Cpmb%7Bu%7D%20%5Cnewline%20%5Cpmb%7BAu%7D%20%3D%20%5Clambda%20%5Cpmb%7Bu%7D)
- two different eigenvectors can have the same eigenvalue
  - an example of [same eigenvalue, different eigenvectors](https://math.stackexchange.com/questions/259017/two-distinct-eigenvectors-corresponding-to-the-same-eigenvalue-are-always-linear)
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D%20%3D%20%5Cbegin%7Bpmatrix%7D0%261%261%5C%5C1%260%261%5C%5C1%261%260%5Cend%7Bpmatrix%7D%20%5Ctextrm%7B%20characteristic%20equation%3A%20%7D%20%5Cbegin%7Bvmatrix%7D%20-%5Clambda%20%26%201%20%26%201%5C%5C%201%20%26%20-%5Clambda%20%26%201%5C%5C%201%20%26%201%20%26%20-%5Clambda%20%5Cend%7Bvmatrix%7D%3D0%5Cnewline%20-%5Clambda%5E3%20&plus;%20%5Clambda%20-%28-%5Clambda%20-%201%29%20&plus;%201%281&plus;%5Clambda%29%20%3D%20-%5Clambda%5E3%20&plus;%203%5Clambda%20&plus;2%20%3D%200%20%5CRightarrow%20%5Clambda%3D-1%20%5Ctextrm%7B%20is%20a%20solution%7D%20%5Cnewline%20%5Ctextrm%7B%20remaining%20equation%20is%3A%20%7D%20-%5Clambda%5E2&plus;%5Clambda&plus;2%20%3D%200%20%5CRightarrow%20%5Clambda%5E2-%5Clambda%20-2%20%3D%200%20%5CRightarrow%20%28%5Clambda-2%29%28%5Clambda&plus;1%29%20%3D%200%20%5Cnewline%20%5Cpmb%7B%5Clambda%7D%20%3D%20%5Cpmb%7B-1%2C-1%2C2%7D%20%5CRightarrow%20%5Cbegin%7Bbmatrix%7D%201%20%26%201%20%26%201%5C%5C%201%20%26%201%20%26%201%5C%5C%201%20%26%201%20%26%201%20%5Cend%7Bbmatrix%7D%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5C%5C%20x_3%20%5Cend%7Bbmatrix%7D%20%5CRightarrow%20x_1%20%3D%20-%28x_2%20&plus;%20x_3%29%20%5Cnewline%20%5Ctextrm%7B%20for%20%7D%20%5Clambda%3D2%20%5C%2C%2C%5C%2C%20%5Cbegin%7Bbmatrix%7D%20-2%20%26%201%20%26%201%5C%5C%201%20%26%20-2%20%26%201%5C%5C%201%20%26%201%20%26%20-2%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20R_1%20%5Crightarrow%20R_1%20-%20R_2%20%5CRightarrow%20%5Cbegin%7Bbmatrix%7D%20-3%20%26%203%20%26%200%20%5C%5C%201%20%26%20-2%20%26%201%5C%5C%201%20%26%201%20%26%20-2%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20R_2%20%5Crightarrow%20R_2%20-%20R_3%20%5CRightarrow%20%5Cbegin%7Bbmatrix%7D%20-3%20%26%203%20%26%200%20%5C%5C%200%20%26%20-3%20%26%203%5C%5C%201%20%26%201%20%26%20-2%20%5Cend%7Bbmatrix%7D)
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20-3%20%26%203%20%26%200%20%5C%5C%200%20%26%20-3%20%26%203%5C%5C%201%20%26%201%20%26%20-2%20%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5C%5C%20x_3%20%5Cend%7Bbmatrix%7D%20%5CRightarrow%20x_1%20%3D%20x_2%20%5C%2C%2C%5C%2C%20x_2%20%3D%20x_3%20%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_3%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%201%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20%5Ctextrm%7B%20the%20corresponding%20unit%20vector%20is%3A%20%7D%20%5Cpmb%7Bv_3%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B3%7D%7D%20%5C%5C%20%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B3%7D%7D%20%5C%5C%20%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B3%7D%7D%20%5Cend%7Bbmatrix%7D%20%5Cnewline%20%5Cpmb%7Bv_1%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20-1%20%5C%5C%200%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_2%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%20-1%20%5C%5C%200%20%5Cend%7Bbmatrix%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7Bv_3%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B3%7D%7D%20%5C%5C%20%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B3%7D%7D%20%5C%5C%20%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B3%7D%7D%20%5Cend%7Bbmatrix%7D)
  - 



## Singular Value Decomposition<a name="svd"></a>





## The Moore-Penrose Pseudoinverse<a name="pseudoinverse"></a>

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%5E%5Cdagger%7D%20%3D%20%5Cpmb%7BVD%5E%7B%5Cdagger%7DU%5ET%7D)
- When ![equation](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D) has more columns than rows(m < n), then solving a linear equation using the pseudoinverse provides one of the many possible solutions. 
  - Specifically, it provides the solution![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BAx%7D%20%3D%20%5Cpmb%7By%7D) with minimal Euclidean norm ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%7C%7C%5Cpmb%7Bx%7D%7C%7C_2) among all possible solutions.
  - <img src="proofs/underdetermined_solution.png" width="500"/>
  - using the pseudoinverse as the inverse of ![equation](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D), ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7Bx%7D%20%3D%20%5Cpmb%7BA%5E%7B-1%7Dy%7D%20%5Capprox%20%5Cpmb%7BA%5E%5Cdagger%20y%7D%20%3D%20%5Cpmb%7BVD%5E%5Cdagger%20U%5ET%20y%7D%20%5Ctextrm%7B%20with%20%7D%20%5Cpmb%7BUU%5ET%7D%20%3D%20%5Cpmb%7BU%5ET%20U%7D%20%3D%20%5Cpmb%7BI%7D_%7Bm%20%5Ctimes%20m%7D%20%5C%2C%2C%5C%2C%20%5Cpmb%7BVV%5ET%7D%20%3D%20%5Cpmb%7BV%5ET%20V%7D%20%3D%20%5Cpmb%7BI%7D_%7Bn%20%5Ctimes%20n%7D%20%5Cnewline%20%5Cpmb%7BD%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Clambda_1%20%26%200%20%26%20%5Ccdots%20%26%200%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%200%20%26%20%5Clambda_2%20%26%20%5Ccdots%20%26%200%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%20%5Cvdots%20%5C%5C%200%20%26%200%20%26%20%5Ccdots%20%26%20%5Clambda_m%20%26%200%20%26%20%5Ccdots%20%26%200%20%5Cend%7Bbmatrix%7D_%7Bm%20%5Ctimes%20n%7D%20%2C%5C%2C%5C%2C%5Cpmb%7BD%5E%5Cdagger%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Clambda_1%5E%7B-1%7D%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%200%20%26%20%5Clambda_2%5E%7B-1%7D%20%26%20%5Ccdots%20%26%200%20%5C%5C%20%5Cvdots%20%5C%5C%200%20%26%200%20%26%20%5Ccdots%20%26%20%5Clambda_m%5E%7B-1%7D%20%5C%5C%200%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%20%5Cvdots%20%5C%5C%200%20%26%200%20%26%20%5Ccdots%20%26%200%20%5Cend%7Bbmatrix%7D_%7Bn%20%5Ctimes%20m%7D%20%5Cpmb%7BD%20D%5E%5Cdagger%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%20%5Ccdots%20%26%200%20%5C%5C%200%20%26%201%20%26%20%5Ccdots%20%26%200%20%5C%5C%20%5Cvdots%20%5C%5C%200%20%26%200%20%26%20%5Ccdots%20%26%201%20%5Cend%7Bbmatrix%7D_%7Bm%20%5Ctimes%20m%7D%20%3D%20%5Cpmb%7BI%7D_%7Bm%20%5Ctimes%20m%7D%20%5Cnewline%20%5Cpmb%7Bx%7D%20%3D%20%5Cpmb%7BA%5E%5Cdagger%20y%7D%20%5CRightarrow%20%5Cpmb%7BAx%7D%20%3D%20%5Cpmb%7BAA%5E%5Cdagger%20y%7D%20%5Cnewline%20%5Cpmb%7BAA%5E%5Cdagger%7D%20%3D%20%5Cpmb%7BUDV%5ET%20VD%5E%5Cdagger%20U%5ET%7D%20%3D%20%5Cpmb%7BUD%20D%5E%5Cdagger%20U%5ET%7D%20%3D%20%5Cpmb%7BU%20U%5ET%7D%20%3D%20%5Cpmb%7BI%7D%20%5Cnewline%20%5Ctherefore%20%5C%2C%2C%5C%2C%20%5Cpmb%7BAx%7D%20%3D%20%5Cpmb%7By%7D)
  - this solution is same as that obtained from the Lagrange multipliers method
    <img src="proofs/underdetermined_pseudoinverse.png" />
  - 
- When ![equation](https://latex.codecogs.com/gif.latex?%5Cpmb%7BA%7D) has more rows than columns(m > n), it is possible for there to be no solution.
  - In this case, using the pseudoinverse gives us the x for which Ax is as close as possible to y in terms of Euclidean norm jjAx − yjj2





[Solving Neural Networks](http://ce.sharif.edu/courses/85-86/2/ce667/resources/root/10%20-%20Ci%20Cho%20Ki/Neura%20Networks%20for%20Solving%20Systems%20of%20Linear.pdf)





## Trace Operator<a name="trace"></a>

- the sum of all elements of the principle diagonal of a matrix
- can be defined for a square or rectangular
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?Tr.%28%5Cpmb%7BA%7D%29%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bmin%28n%2Cm%29%7D%20A_%7Bii%7D)
- Frobenius norm is the square root of trace of ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cpmb%7BAA%5ET%7D)
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%20the%20general%20term%20of%20%7D%20%5Cpmb%7BAA%5ET%7D%20%3D%20B_%7Bij%7D%20%3D%20%5Csum%5Climits_%7Bk%3D1%7D%5En%20A_%7Bik%7DA_%7Bkj%7D%27%20%5Ctextrm%7B%20such%20that%20%7D%20A%27_%7Bkj%7D%20%3D%20A_%7Bjk%7D%20%5Cnewline%20Tr.%28%5Cpmb%7BAA%5ET%7D%29%20%3D%20%5Csum%5Climits_%7Bi%3D1%2Cj%3Di%7D%5En%20B_%7Bij%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%2Cj%3Di%7D%5En%20%5Csum%5Climits_%7Bk%3D1%7D%5En%20A_%7Bik%7DA_%7Bik%7D%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5Csum%5Climits_%7Bk%3D1%7D%5En%20A_%7Bik%7D%5E2%20%3D%20%7C%7C%5Cpmb%7BA%7D%7C%7C_F%5E2%20%5Cnewline%20%5Ctherefore%20%5C%2C%2C%5C%2C%20%7B%5Ccolor%7Bred%7D%20%7C%7C%5Cpmb%7BA%7D%7C%7C_F%20%3D%20%5Csqrt%7BTr.%28%5Cpmb%7BAA%5ET%7D%29%7D%7D)





# Principal Component Analysis

- x
- <img src="proofs/trace_max_d_1.png" />
- 