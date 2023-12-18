# [**Probabilistic Deep Learning and Transfer Learning for Robust Cryptocurrency Price Prediction under Uncertainty**](https://www.researchgate.net/publication/373271371_Probabilistic_Deep_Learning_and_Transfer_Learning_for_Robust_Cryptocurrency_Price_Prediction_under_Uncertainty)

**Authors:**

**Amin Golnari <sup>a<sup>**, **Mohammad Hossein Komeili <sup>b<sup>**, **Zahra Azizi <sup>c<sup>** <br>
a) Faculty of Electrical & Robotics, Shahrood University of Technology, Shahrood, Iran <br>
b) Faculty of Mathematics & Computer Science, Shahid Beheshti University, Tehran, Iran <br>
c) Department of Computer Engineering, University of Afarinesh, Borujerd, Iran

**Link to the preprint version on ResearchGate:** [Click here](https://www.researchgate.net/publication/373271371_Probabilistic_Deep_Learning_and_Transfer_Learning_for_Robust_Cryptocurrency_Price_Prediction_under_Uncertainty)

**Or You Can Run this Python Code [Demo] on Google Colab:**    

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amingolnari/Demo-BTCUSD-PricePred-ProbabilisticDL-TransferLearning/blob/main/Demo_BTCUSD_PricePred_5min_TensorFlowProb.ipynb)

*Cryptocurrency Price Prediction Demo using Probabilistic Deep Learning and Transfer Learning*


**This demo showcases the prediction of cryptocurrency prices using probabilistic deep learning models. Various architectures based on Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) are implemented for this purpose. Additionally, the code demonstrates transfer learning, where a pre-trained model on Bitcoin (BTC) price data is used as a foundation to train models for other cryptocurrencies.**

**Models Utilized:**
- *Bidirectional Probabilistic GRU (bi_gru_prob)*
- *Bidirectional Simple GRU (bi_gru_simple)*
- *Bidirectional GRU with Time-Distributed Dense (bi_gru_time_dist)*
- *Probabilistic GRU (gru_prob)*
- *Simple GRU (gru_simple)*
- *GRU with Time-Distributed Dense (gru_time_dist)*
- *Bidirectional Probabilistic LSTM (bi_lstm_prob)*
- *Bidirectional Simple LSTM (bi_lstm_simple)*
- *Bidirectional LSTM with Time-Distributed Dense (bi_lstm_time_dist)*
- *Probabilistic LSTM (lstm_prob)*
- *Simple LSTM (lstm_simple)*
- *LSTM with Time-Distributed Dense (lstm_time_dist)*


**Workflow Overview:**
1. **Data Preprocessing:**
   - *Download cryptocurrency price data, e.g., Bitcoin (BTC), using Yahoo Finance.*
   - *Normalize and split the data into training and testing sets.*

2. **Model Training:**
   - *Train various deep learning models (including probabilistic models) with different architectures on BTC price data.*
   - *Models include both GRU and LSTM variants.*

3. **Transfer Learning:**
   - *Utilize the best-performing model (e.g., gru_prob) as a pre-trained model.*
   - *Transfer this model to predict prices for other cryptocurrencies.*

4. **Evaluation and Analysis:**
   - *Evaluate model performance using metrics such as R2 score, Mean Absolute Percentage Error, and more.*
   - *Generate Residuals vs. Predicted Values plots for each model.*

5. **Reporting and Visualization:**
   - *Report and visualize the results, including prediction plots for different models.*
   - *Provide insights into model performance and potential use in predicting other cryptocurrency prices.*

**Note:** Ensure proper installation of required libraries, including TensorFlow Probability and yfinance.


**Note:** This innovative methodology challenges the conventional approach to model selection by recognizing that optimal performance may not align with epochs characterized by the absolute minimum value of the loss function on the validation dataset. Traditionally, the tendency has been to associate the best model with the epoch where the loss function achieves its minimum value. However, our approach introduces a more nuanced perspective, considering scenarios where a model at epoch 20 with a loss function value outperforms its minimum loss at epoch 30 on the validation dataset. This paradigm shift redirects the focus from fixating solely on minimum loss values to a dynamic assessment that accounts for the model's efficacy at different epochs. R2-score monitoring on the validation dataset becomes pivotal in identifying epochs where the model excels in capturing underlying patterns in cryptocurrency price data. This adaptive and forward-looking approach ensures a more nuanced and resilient model selection, enhancing the robustness of cryptocurrency price prediction models without being confined to the traditional emphasis on achieving the lowest loss function value.
