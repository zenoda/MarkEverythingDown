# Basic Data Types in R: Numeric

## Numeric: Default Data Type in R Representing Decimal Values

- **Numeric:** The default data type in R for representing decimal values.
  - Assign a decimal value:
    ```R
    x <- 3.14
    ```
  - Print the value of `x`:
    ```R
    x
    # [1] 3.14
    ```
  - Print the class name of `x`:
    ```R
    class(x)
    # [1] "numeric"
    ```
  - Assign an integer value:
    ```R
    k <- 3
    ```
  - Print the value of `k`:
    ```R
    k
    # [1] 3
    ```
  - Print the class name of `k`:
    ```R
    class(k)
    # [1] "numeric"
    ```
- Even integer values are stored as numeric unless explicitly declared:
    ```R
    class(k)
    # [1] "numeric"
    ```
  - Check if `k` is an integer:
    ```R
    is.integer(k)
    # [1] FALSE
    ```

## Try it Yourself:
- [Link to Practice](https://campus.datacamp.com/courses/r-short-and-sweet/hello-r?ex=2)

