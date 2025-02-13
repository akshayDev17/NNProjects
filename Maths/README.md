# Vector identities
- single dimensional, not 2d where the other dimension = 1

## Dot product is commutative
- for vector x $x^T = x$ 
- hence, $x^Ty = (x^Ty)^T = y^Tx$

# Matrix Identities

- Matrix representation: $\mathbf{A} \, \epsilon \, \mathbb{R}^{m \times n}$

## $(AB)^T = B^TA^T$
- $A: m_a \times n \,,\, B: n\times m_b$
- $(AB)_{ij} = \sum\limits_{k=1}^{n} A_{ik}.B_{kj} \Rightarrow (AB)^T_{ij} = (AB)_{ji} = \sum\limits_{k=1}^{n} A_{jk}.B_{ki}$
- $(B^TA^T)_{ij} = \sum\limits_{k=1}^{n} B^T_{ik}.A^T_{kj} = \sum\limits_{k=1}^{n} B_{ki}.A_{jk}$
- $\therefore \,\,,\, (B^TA^T)_{ij} = (AB)^T_{ij} \Rightarrow \mathbf{(AB)^T = B^TA^T}$

## rank(AB) $\le$ min.(rank(A), rank(B))
- $A_{m \ times n} \,,\, B_{n\times p} \Rightarrow AB_{m \times p}$.
- the product will have its columns as linear combination of columns of $A$
    - hence the max no. of linearly independent columns(its column rank) will never exceed that of $A$
- the product will have its rows as linear combination of rows of $B$
    - hence the max no. of linearly independent rows(its row rank) will never exceed that of $B$
- additionally, w.r.t. the columns of $AB$
    - each column of $AB$ is a linear combination of columns of $A$, where the weights are the corresponding column elements of B
    - <img src="images/Column rank of AB due to B-1.jpg" width=400 /><img src="images/Column rank of AB due to B-2.jpg" width=400/>
    - if B has some linearly dependent columns, when such a column is used as weights for creating the corresponding column in $AB$
        - this corresponding column of $AB$ can easily be expressed as the linear combination of some other columns of $AB$ that were formed from linearly independent columns of both $A$ and $B$
    - hence, the column rank of $AB$ gets limited by that of $B$ as well.
- this is the fundamental principle behind LoRA optimisation in LLMs

# Mathematical Relations (Functions)
- $\mathcal{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$
- this can be thought of as an n-dimensional vector being transformed to an m-dimensional vector: $x \,\epsilon\, \mathbb{R}^n \rightarrow b \,\epsilon\, \mathbb{R}^m$
    - a matrix of dimensionality $m \times n$ will be used for this transformation s.t. $Ax = b$

# Linear Dependence and Span

## Span
- for a given set of vectors $V = {v_1, v_2\cdots v_n}$, span = linear combination of these vectors, i.e. $\sum\limits_{i=1}^n c_i.v_i$

## Column-space
- A in terms of column-vectors is defined as $A = \begin{bmatrix} | & | & \cdots & | \\ A_{:,1} & A_{:,2} & \cdots & A_{:,n} \\ | & | & \cdots & | \end{bmatrix}_{m\times n}$
- column space of A is basically a linear combination of these $A_{:, i}$ terms, i.e. span of A's columns.
- **column-space of A** = $\sum\limits_{i=1}^n c_i.A_{:, i}$
- the system of linear equations represented in the form $\mathbf{Ax} = \mathbf{b}$ , where $A$ and $b$ are known, and $x$ is to be found out can be represented as 
    $b = \sum\limits_{i=1}^m x_i.A_{:, i}$, i.e. **checking** if $b$ is in **column-space of A**.

## Linear Dependence
- a set of vectors $V = {v_1, v_2\cdots v_n}$ is said to be linearly independent if no vector $v_j$ can be expressed a linear combination of the remaining vectors $V_{-v_j}$ (set of vectors in $V$ except $v_j$)
- for an $\mathbb{R}^{m}$ dimensional vector space, there can be no more than m-linearly independent vectors
    - for instance, take this example of $\mathbb{R}^2$: <img src="images/N greater than m linear dependence.jpg" width=400 />
- conversely, a vector is said to be linearly dependent when it can be expressed as a linear combination, i.e. a span of some linearly independent vectors.
    - in the above example, the $3^{rd}$ vector was a linearly dependent vector.
- in the context of the system of linear equations $\mathbf{Ax} = \mathbf{b}$, 
    - since $b \, \epsilon \, \mathbb{R}^m$, we place expectations on the transformation matrix $A$.
    - **the first expectation** is that $A$ should have at least `m` columns, i.e. $\mathbf{n \ge m}$, or else the column span of A won't be equal to $\mathbb{R}^m$
        - take this as an example <img src="images/Linear independence and dimensionality.jpg" height=300/>
            - in this, only 2 lines will be enough to *span* the XY plane , i.e. $\mathbb{R}^2$
        - for $n < m$ it is definitionally impossible to have a set of vectors of size n , all linearly independent such that they are able to span the m-dimensional space (a higher dimensional space)
            - take the example $n = 2, m = 3$, then any possible set of 2D vectors will only be able to span a particular plane, and not the entire 3D space, but $\mathbb{R}^3$ is the 3D space.
            - mathematically:
                - $\mathbb{S}={v_1, v_2 \cdots v_n}$ be a set of n vectors in $\mathbb{R}^m$ with $n < m$
                - However, spanning a vector space means that the span of $\mathbb{S}$ must have dimension at least $m$. 
                    - But the maximum number of linearly independent vectors in $\mathbb{S}$ is at most n.
                    - This implies that the subspace spanned by $\mathbb{S}$ has dimension at most $n$ , which is strictly less than $m$. (i.e. $\mathbb{S}$ spans $\mathbb{R}^n$)
                - But wait! We assumed that $\mathbb{S}$ spans $\mathbb{R}^m$, which means that span($\mathbb{S}$) should have dimension exactly $m$.
                - This contradiction shows that $\mathbb{S}$ cannot span $\mathbb{R}^m$.
        - **Note:** $\mathbf{n \ge m}$ doesn't guarantee existence of a solution, it rather guarantees that a solution could be found for **any and all possible values of** $\mathbf{b}$
    - **the second expectation** is that $A$ should have **at least 1 set** of **`m` linearly independent columns**
        - a span of this `m` linearly independent columns, i.e. `m` linearly independent m-dimensional vectors will have a span equal to $\mathbb{R}^m$ (Because dimension is the maximal number of linearly independent vectors—by its very construction, and this set of linearly independent vectors that *span* the vector space is called the **basis**.)
        - $A$ can have multiple groups of linearly independent columns, each group with varying sizes, but to solve the system of equations it should have at least 1 m-sized group.
    - **the third expectation** is that $A$ should **have a true inverse**
        - $\because \, Ax = b \Rightarrow x = A^{-1}b$. this necessitates the inverse requirement.
        - for the system of linear equations that usually represent some real world process, the transformation function (A) should be **injective and surjective** (one-to-one and onto mapping, i.e. the mapping covers the whole co-domain)
        - for instance, consider n = 3 and m = 2, 
            - there are infinitely many 3D points that can be mapped to the same 2D point, 
            - but the inverse relation has no reproducibility because it wouldn't know which specific 3D point should be mapped to a given 2D point (inverse relation will have inputs and outputs inverted)
            This is called the **non-trivial null space** because at least 1 non-null vector resides in the null space.
        - mathematically
            - if $n > m$ , $Ax = b$ will have a null space where $Ax_{null}=0 \,\, (x_{null} \textrm{ is not a null vector })$ , and thus for unique solutions $x_p$ s.t. $Ax_p = b$, we end up having infinitely many solutions of type $x = x_{null} + x_p$
            - 
        - hence if $\mathbf{A}$ is to be injective, then $\mathbf{n \le m}$
        - hence, $A$'s injectivity requirement necessitates the requirement of a unique inverse.
        - this corresponds to having **at most `m` columns**.
    - since $n \le m \textrm{ and } n \ge m \Rightarrow n = m$
    - hence, **for the transformation** $\mathbf{A}$ to be **injective and surjective**, it should be a **square matrix**
