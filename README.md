# Privacy-preserving SVM assuming public model private data scenario

Implementation of privacy-preserving SVM assuming public model private data scenario (data in encrypted but model parameters are unencrypted) using adequate partial homomorphic encryption

### System Design

![System-Design](https://github.com/abhinav-bohra/Privacy-Preserving-ML/blob/main/system_design.png)

1. **Train SVM Classifier model on server using public data:** In this step, a support vector machine (SVM) model is trained on the server using public data. The public data is data that is available to everyone and does not contain any sensitive information. This model will be used to make predictions on the encrypted data sent by the client.
2. **Encrypt the data on the client:** The client encrypts their private data (X_test) using a homomorphic encryption scheme. Homomorphic encryption allows computations to be performed on encrypted data without decrypting it first, preserving the privacy of the data. The encryption process generates a ciphertext that is sent to the server for prediction.
3. **Send the encrypted X_test to the server:** The encrypted data is sent to the server for prediction. The server can perform computations required for inference on the encrypted data without having access to the original plaintext.
4. **Use unencrypted model parameters for inference:** The server uses the unencrypted model parameters to perform the prediction on the encrypted data. The model parameters are not encrypted and can be used directly for prediction because inference operations required simple addition and multiplication between one encrypted and one unencrypted number
5. **Send model predictions back to the client:** After the prediction is performed, the server sends the encrypted predictions (Y_pred) back to the client.
6. **On the client, decrypt Y_pred and calculate accuracy:** The client decrypts the encrypted predictions (Y_pred) using the private key to obtain the final predictions in plaintext. The accuracy of the predictions can be calculated by comparing the predicted labels with the actual labels of the test data (Y_test)

Partial homomorphism in encryption allows for some limited computations to be performed on encrypted data. In the context of privacy-preserving SVM, it enables the server to perform the prediction on the encrypted data while preserving the privacy of the client's data.

#### Data-set used:

This example involves learning using sensitive medical data from multiple hospitals to predict diabetes progression in patients. The data is a standard dataset from sklearn
- Train-set size: 614 samples
- Test set size: 154 samples
- Number of Features: 8
- Target Outcome: 0 or 1 (Binary Classification)

#### Timing Details:
- Time taken by normal SVM model = 3.31 secs
- Time taken by privacy preserving SVM model = 33.56 secs

------------

This project was completed as part of the programming assignment for the course **Dependable and Secure AI-ML (AI60006).**
