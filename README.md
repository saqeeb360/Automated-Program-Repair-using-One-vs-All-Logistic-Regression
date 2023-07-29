# Predicting-Error-Types-in-Code-with-Token-Based-Machine-Learning
Automated Program Repair using One-vs-All Logistic Regression is a machine learning approach aimed at solving the challenging problem of identifying and repairing errors in computer programs. Leveraging the power of one-vs-all logistic regression, this method enables the classification of program code lines into multiple error classes, making it invaluable for educational purposes and reducing the burden on teaching assistants.

The core of this approach lies in transforming each code line into a bag-of-words (BoW) representation, capturing the occurrence count of tokens. Then, a one-vs-all logistic regression model is trained for each error class, treating it as a binary classification problem. By training multiple models, the approach can effectively handle multi-class classification and predict the most probable error class for a given code line.

The resulting model is capable of suggesting repairs to students whose code fails to compile, enhancing the learning experience by providing tailored feedback on potential errors. Additionally, the model's ability to output multiple ranked error classes allows for better insights into the possible issues in the code, enabling more efficient debugging and problem-solving.

Overall, Automated Program Repair using One-vs-All Logistic Regression is a valuable tool that brings pedagogical value to programming students, streamlines the teaching process, and fosters an enjoyable learning experience by automating the error identification and repair process.

---
##### Few Tokens given a number 
.
| **Number** | **Token**                |
|------------|--------------------------|
| 0          | !                        |
| 1          | !=                       |
| 106        | TokenKind.LITERAL_CHAR   |
| 107        | TokenKind.LITERAL_DOUBLE |
| 220        | while                    |
| 221        | {                        |

##### Few Error codes with their respective ids
.
| **Id** | **Error message**               |
|--------|---------------------------------|
| 1      | expected ID                     |
| 2      | expected ID after expression    |
| 3      | use of undeclared identifier ID |
| 4      | expected expression             |
| 5      | expected identifier or ID       |

# Results
prec@1: 0.813 prec@3: 0.947 prec@5: 0.974
mprec@1: 0.815 mprec@3: 0.948 mprec@5: 0.971