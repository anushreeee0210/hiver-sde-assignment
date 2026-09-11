# AppleSupport Customer Support Reply System

An NLP-based customer support reply system built for the Hiver SDE
take-home assignment.

The system uses the public Customer Support on Twitter dataset, extracts
conversations involving AppleSupport, identifies the customer's support
intent, retrieves a relevant historical AppleSupport response using
TF-IDF similarity and keyword-based reranking, and exposes the result
through a FastAPI REST API.

## Features

-   Extracts AppleSupport conversations from the Twitter
    customer-support dataset.
-   Builds direct customer-message to AppleSupport-response pairs.
-   Uses a 10-intent support taxonomy.
-   Creates a 200-example golden evaluation set.
-   Uses TF-IDF with unigram and bigram features.
-   Uses cosine similarity for retrieval.
-   Uses keyword/domain reranking to improve relevance.
-   Cleans Twitter mentions from retrieved responses.
-   Returns grounded support guidance from similar historical
    AppleSupport cases.
-   Provides a FastAPI REST API with Swagger documentation.

## Architecture

``` text
Customer Query
      |
      v
Intent Detection
      |
      v
TF-IDF Retrieval
      |
      v
Candidate Retrieval
      |
      v
Keyword / Domain Reranking
      |
      v
Best Historical Support Response
      |
      v
Response Cleaning
      |
      v
FastAPI JSON Response
```

## Dataset

Source: Customer Support on Twitter by Thought Vector.

Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The original dataset contains approximately 2.8 million tweets and
fields including `tweet_id`, `author_id`, `inbound`, `created_at`,
`text`, `response_tweet_id`, and `in_response_to_tweet_id`.

This project filters the dataset to AppleSupport conversations and
creates direct customer-to-support pairs.

### Data processing

1.  Extract AppleSupport responses.
2.  Find tweets related to those responses.
3.  Identify customer messages.
4.  Match each customer message with the corresponding AppleSupport
    response.
5.  Remove duplicate/invalid/very short retrieval examples.
6.  Build the retrieval index.

## Intent Taxonomy

  -----------------------------------------------------------------------
  Intent                              Description
  ----------------------------------- -----------------------------------
  `software_ios`                      iOS updates, software problems,
                                      freezing and lagging

  `app_issues`                        Problems with applications and app
                                      behavior

  `battery_charging`                  Battery drain, charging and charger
                                      issues

  `hardware_device`                   Screen, camera, keyboard, speaker
                                      and device hardware

  `wifi_connectivity`                 Wi-Fi, wireless, internet and
                                      network issues

  `icloud_apple_id`                   iCloud, Apple ID, passwords and
                                      login/account issues

  `app_store_purchases`               App Store, iTunes, purchases,
                                      refunds, payments and subscriptions

  `mac_macos`                         MacBook, iMac, macOS and Finder
                                      issues

  `notifications_messages`            Notifications, Messages and
                                      iMessage issues

  `security_fraud`                    Phishing, scams, suspicious emails
                                      and security issues
  -----------------------------------------------------------------------

## Golden Evaluation Set

`data/golden_set.csv` contains 200 examples, with 20 examples per
intent.

The current labels were created using keyword-assisted labeling and
should be considered a baseline rather than a fully human-validated
benchmark.

## Retrieval

The retrieval system uses:

### TF-IDF

-   Unigrams and bigrams
-   Up to 100,000 features

### Cosine similarity

The customer query is compared against historical AppleSupport customer
messages.

### Keyword reranking

Retrieved candidates are reranked using important domain terms such as:

``` text
battery
wifi
icloud
refund
purchase
payment
app
ios
update
macbook
notifications
screen
camera
```

This helps avoid generic keyword matches. For example, an App Store
refund query should prefer an App Store/refund case over an unrelated
laptop refund case.

## Response Generation

The system selects the most relevant historical AppleSupport response
and cleans it before returning it.

Cleaning includes:

-   Removing Twitter `@mentions`
-   Normalizing whitespace
-   Preserving useful support guidance
-   Avoiding unsupported troubleshooting instructions

The response is therefore grounded in the historical support dataset.

# Project Structure

``` text
hiver-sde-assignment/
|
├── data/
│   ├── twcs.csv
│   ├── applesupport_raw.csv
│   ├── applesupport_conversations.csv
│   ├── applesupport_pairs.csv
│   ├── golden_set.csv
│   └── retriever_evaluation.csv
|
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── retriever.pkl
│   ├── retrieval_data.pkl
│   ├── intent_vectorizer.pkl
│   └── intent_classifier.pkl
|
├── src/
│   ├── extract_applesupport.py
│   ├── inspect_applesupport.py
│   ├── find_applesupport_id.py
│   ├── extract_applesupport_conversations.py
│   ├── analyze_conversations.py
│   ├── create_support_pairs.py
│   ├── discover_intents.py
│   ├── topic_analysis.py
│   ├── create_golden_set.py
│   ├── build_retriever.py
│   ├── test_retriever.py
│   ├── train_intent_classifier.py
│   ├── test_intent_classifier.py
│   ├── intent_retriever.py
│   ├── evaluate_retriever.py
│   ├── analyze_errors.py
│   ├── generate_response.py
│   └── api.py
|
├── requirements.txt
├── README.md
└── .gitignore
```

# Setup

## 1. Clone the repository

``` powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd hiver-sde-assignment
```

## 2. Create a virtual environment

``` powershell
python -m venv venv
```

## 3. Activate the environment

Windows PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

``` powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

``` powershell
.\venv\Scripts\Activate.ps1
```

## 4. Install dependencies

``` powershell
pip install -r requirements.txt
```

If required:

``` powershell
pip install fastapi uvicorn
```

# Dataset Setup

Download the Customer Support on Twitter dataset from Kaggle and place
the original CSV at:

``` text
data/twcs.csv
```

Do not commit this large raw dataset to GitHub.

# Running the Data Pipeline

If the processed files and models are already present, you do not need
to rerun the complete pipeline.

For a fresh setup:

``` powershell
python src/extract_applesupport.py
python src/extract_applesupport_conversations.py
python src/create_support_pairs.py
python src/topic_analysis.py
python src/create_golden_set.py
python src/build_retriever.py
python src/train_intent_classifier.py
python src/evaluate_retriever.py
python src/analyze_errors.py
```

# Test the Response Generator

Run:

``` powershell
python src/generate_response.py
```

Enter a customer query.

## Example 1: Battery

``` text
My iPhone battery is draining very quickly
```

Expected intent:

``` text
battery_charging
```

## Example 2: Wi-Fi

``` text
My WiFi keeps disconnecting on my iPhone
```

Expected intent:

``` text
wifi_connectivity
```

## Example 3: iCloud

``` text
I can't log into my iCloud account
```

Expected intent:

``` text
icloud_apple_id
```

## Example 4: App Store refund

``` text
I want a refund for an App Store purchase
```

Expected intent:

``` text
app_store_purchases
```

## Example 5: Mac

``` text
My MacBook is running extremely slowly after the latest update
```

Expected intent:

``` text
mac_macos
```

## Example 6: Messages

``` text
My iPhone isn't showing any notifications for new messages
```

Expected intent:

``` text
notifications_messages
```

# FastAPI

The project exposes the support system as a REST API.

## Start the API

From the project root:

``` powershell
python -m uvicorn src.api:app --reload
```

Server:

``` text
http://127.0.0.1:8000
```

## Check the API

Open:

``` text
http://127.0.0.1:8000
```

Expected response:

``` json
{
  "message": "AppleSupport Customer Support API",
  "status": "running"
}
```

## Swagger Documentation

Open:

``` text
http://127.0.0.1:8000/docs
```

Find:

``` text
POST /reply
```

Click `Try it out`, enter a query, and click `Execute`.

## API Request

``` json
{
  "query": "I want a refund for an App Store purchase"
}
```

## API Response

Example:

``` json
{
  "query": "I want a refund for an App Store purchase",
  "intent": "app_store_purchases",
  "response": "We'd like to get you pointed in the right direction. You'll want to contact our iTunes Advisors: https://t.co/SDIe7UiyJN",
  "similarity": 0.3536,
  "retrieved_issue": "Just got conned into an app store purchase which didnt give me an option to say no. How do I get a refund?"
}
```

The exact historical response and similarity score may vary.

# Evaluation

A 200-example golden set was used for baseline intent-routing
evaluation.

Current baseline:

``` text
Total examples: 200
Correct: 142
Incorrect: 58
Intent accuracy: 71.00%
```

The errors mainly occur where support topics overlap, including:

-   iOS updates vs application problems
-   hardware vs general device issues
-   iCloud vs account/security issues
-   notifications vs software updates

The final system therefore uses intent detection as a routing signal and
relies on grounded retrieval plus reranking for response selection.

# End-to-End Example

Input:

``` text
I want a refund for an App Store purchase
```

Flow:

``` text
1. Detect intent
   -> app_store_purchases

2. Retrieve similar AppleSupport cases

3. Rerank candidates using semantic similarity
   and important domain keywords

4. Select the best historical support response

5. Remove Twitter mentions

6. Return the grounded response through the API
```

# Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   TF-IDF
-   Cosine Similarity
-   Nearest Neighbors
-   Logistic Regression
-   FastAPI
-   Uvicorn
-   Pickle
-   PowerShell

# Design Decisions

## Why retrieval?

The dataset contains real AppleSupport responses. Retrieving similar
historical cases keeps responses grounded and reduces unsupported
recommendations.

## Why TF-IDF?

TF-IDF is lightweight, fast, interpretable, and suitable for
keyword-heavy customer-support queries.

## Why keyword reranking?

Generic words such as `refund`, `problem`, and `help` can produce poor
matches. Domain-specific keyword overlap improves matching for areas
such as App Store, iCloud, Wi-Fi, battery and Mac support.

# Limitations

1.  The intent taxonomy is manually defined from topic analysis.
2.  The golden labels are keyword-assisted and should be human-reviewed
    for production use.
3.  TF-IDF is less effective than modern embedding models for heavily
    paraphrased queries.
4.  Historical Twitter responses can be short or generic.
5.  Some historical responses contain external support links.
6.  The system retrieves grounded historical guidance instead of
    generating completely novel troubleshooting instructions.
7.  Some support categories overlap.

# Future Improvements

-   Replace TF-IDF with sentence embeddings.
-   Add a cross-encoder reranker.
-   Human-validate the complete golden set.
-   Add Recall@K, MRR and Precision@K metrics.
-   Add confidence thresholds and fallback handling.
-   Use an LLM to rewrite retrieved responses while keeping them
    grounded.
-   Add conversation history.
-   Add response safety/quality filters.
-   Add a frontend dashboard.
-   Containerize the API with Docker.

# Quick Start

For an already-built project:

``` powershell
cd "C:\Games\Code\Z Project\hiver-sde-assignment"
.\venv\Scripts\Activate.ps1
python -m uvicorn src.api:app --reload
```

Then open:

``` text
http://127.0.0.1:8000/docs
```

Use:

``` json
{
  "query": "My iPhone battery is draining very quickly"
}
```

Execute `POST /reply`.

# Author

Developed as part of the Hiver SDE Take-Home Assignment.
