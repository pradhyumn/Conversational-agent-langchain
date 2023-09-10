# Conversational-agent-langchain

This is a toy-example for creating a Python application that acts as a conversational agent based on prompts to an uploaded PDF. The application uses Open AI's API calls to use GPT-3 (or GPT 4 if you have the subscription) to feed the text to the LLM using langchains. This toy-example was created to answer ESG (Environmental, Social and Governance) based queries from an uploaded PDF, but can be easily extended to incorporate any use cases by just updating the prompts appropriately.

### Installation

First step is to clone the repository and install the requirements mentioned in the requirements.txt file. You will also need to add your OPENAI API key to the ```.env``` file.

### Usage

To use the application, run the ```app.py``` file using the following command:

```
streamlit run app.py
```

This will run a streamlit application hosted locally and can be run on your browser. In the browser window, upload your PDF, followed by the appropriate prompts for the queries you want the LLM model to answer from the PDF. Some example prompts for the Governance-related queries are present in the ``` Governance.txt``` file.

### Lambda Function

The ```lambda_function.py``` file contains the code for deploying the application to AWS Lambda instead of hosting it locally. The ```test.py ``` file contains the test cases for automatically testing the pdf based on prompts on the deployed model using a post call using Postman.

### Contributing

This repository is for educational purposes only.
