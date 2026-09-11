# AppleSupport Customer Support Reply System

An NLP-based customer-support reply system built for the **Hiver SDE
Take-Home Assignment**.

The system focuses on the **AppleSupport** brand from the Customer
Support on Twitter dataset. It extracts customer → support
conversations, identifies the support topic, retrieves similar
historical AppleSupport cases using TF-IDF, reranks candidates using
domain-specific lexical signals, and exposes the final grounded reply
through a FastAPI API.

------------------------------------------------------------------------

# 1. Problem Framing

## What does "good" mean for AppleSupport?

For this project, a good support reply should:

1.  **Address the customer's actual issue** rather than just matching
    generic words.
2.  **Be grounded in historical AppleSupport guidance** from the
    dataset.
3.  **Avoid hallucinating troubleshooting steps** that are not supported
    by the retrieved evidence.
4.  **Be concise and professional**, similar to real customer-support
    interactions.
5.  **Route different Apple support problems correctly**, such as
    battery, Wi-Fi, iCloud, App Store purchases, Mac, and Messages.
6.  **Handle noisy social-media language**, including abbreviations,
    spelling variations, mentions, URLs, and short messages.

## What I chose not to build

To keep the scope focused on the assignment, this project does not
attempt to build:

-   A complete Apple customer-service platform.
-   A production authentication system.
-   A live Apple support integration.
-   Automated refunds, purchases, account changes, or other real-world
    actions.
-   A fully autonomous LLM agent.
-   A production-scale vector database.
-   A frontend dashboard.
-   A system that invents troubleshooting instructions without evidence.

The goal is a **grounded support-reply retrieval system**, not a
replacement for Apple's complete support infrastructure.

------------------------------------------------------------------------

# 2. Approach

``` text
                 Customer Query
                       |
                       v
                Text Cleaning
                       |
                       v
                Intent Routing
                       |
                       v
               TF-IDF Retrieval
                       |
                       v
              Candidate Responses
                       |
                       v
             Domain Keyword Reranking
                       |
                       v
              Best Historical Case
                       |
                       v
             Response Cleaning
                       |
                       v
                 FastAPI API
```

The main design principle is:

> Retrieve evidence first, then return a response grounded in that
> evidence.

This reduces the risk of generating unsupported advice.

------------------------------------------------------------------------

# 3. Dataset

Dataset:

**Customer Support on Twitter** by Thought Vector

Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The original dataset contains approximately 2.8 million tweets and
includes:

-   `tweet_id`
-   `author_id`
-   `inbound`
-   `created_at`
-   `text`
-   `response_tweet_id`
-   `in_response_to_tweet_id`

The project filters the dataset to conversations involving:

``` text
AppleSupport
```

The original raw dataset should be downloaded separately and placed at:

``` text
data/twcs.csv
```

Because the raw dataset is large, it should not be committed to GitHub.

------------------------------------------------------------------------

# 4. Data Preparation

The preprocessing pipeline performs the following:

### Step 1 --- Extract AppleSupport responses

``` powershell
python src/extract_applesupport.py
```

Creates:

``` text
data/applesupport_raw.csv
```

### Step 2 --- Extract related conversations

``` powershell
python src/extract_applesupport_conversations.py
```

Creates:

``` text
data/applesupport_conversations.csv
```

### Step 3 --- Build customer → AppleSupport pairs

``` powershell
python src/create_support_pairs.py
```

Creates:

``` text
data/applesupport_pairs.csv
```

The resulting dataset contains direct pairs such as:

``` text
Customer:
My wifi keeps disconnecting on my phone.

AppleSupport:
We'd be happy to help you out...
```

------------------------------------------------------------------------

# 5. Intent Taxonomy

The project uses 10 support intents discovered from the AppleSupport
conversations:

  -----------------------------------------------------------------------
  Intent                              Description
  ----------------------------------- -----------------------------------
  `software_ios`                      iOS updates, software problems,
                                      freezing and lagging

  `app_issues`                        Application-related problems

  `battery_charging`                  Battery drain, charging and charger
                                      issues

  `hardware_device`                   Screen, camera, keyboard, speaker
                                      and hardware issues

  `wifi_connectivity`                 Wi-Fi, wireless, internet and
                                      network issues

  `icloud_apple_id`                   iCloud, Apple ID, passwords and
                                      login issues

  `app_store_purchases`               App Store, iTunes, purchases,
                                      refunds, payments and subscriptions

  `mac_macos`                         MacBook, iMac, macOS and Finder
                                      issues

  `notifications_messages`            Notifications, Messages and
                                      iMessage problems

  `security_fraud`                    Phishing, scams, suspicious emails
                                      and security issues
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 6. Golden Evaluation Set

The project contains:

``` text
data/golden_set.csv
```

with **200 examples**, which is within the required 150--250 example
range.

The examples were sampled from AppleSupport customer messages and
distributed across the 10 support intents, with approximately 20
examples per intent.

## Sampling

Examples were selected from the processed AppleSupport customer/support
pairs after basic cleaning and duplicate removal.

The sampling was designed to cover common support areas including:

-   Battery
-   Wi-Fi
-   iCloud
-   App Store purchases
-   iOS
-   Apps
-   Mac
-   Hardware
-   Notifications
-   Security

## Labelling

The current version of the set was initially created using
**keyword-assisted labeling** based on the intent taxonomy and topic
analysis.

Important:

> The current `golden_set.csv` should be treated as a **draft evaluation
> set until all 200 labels have been manually reviewed**. It would be
> misleading to describe the existing file as fully hand-labelled
> without completing that review.

For a final submission, the recommended process is:

1.  Randomly sample 150--250 examples.
2.  Hide the automatically predicted label.
3.  Manually assign exactly one intent.
4.  Review ambiguous examples a second time.
5.  Freeze the final CSV.
6.  Never use the golden set to train the final retrieval model.

This prevents evaluation leakage.

------------------------------------------------------------------------

# 7. Retrieval Model

The main retrieval model uses:

## TF-IDF

Configuration:

-   Unigrams
-   Bigrams
-   Up to 100,000 features

## Cosine similarity

The customer query is transformed into a TF-IDF vector and compared with
historical AppleSupport customer messages.

## Keyword reranking

The top retrieved candidates are reranked using important domain terms
such as:

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

This prevents generic word matches from dominating the result.

For example:

``` text
I want a refund for an App Store purchase
```

should retrieve an App Store/refund case rather than:

``` text
I want a refund for my laptop
```

------------------------------------------------------------------------

# 8. Response Generation

The system does not invent unsupported troubleshooting steps.

Instead, it:

1.  Detects the likely support intent.
2.  Retrieves similar historical AppleSupport cases.
3.  Reranks the candidates.
4.  Selects the strongest historical support response.
5.  Cleans Twitter handles and formatting.
6.  Returns the grounded response.

Example:

``` text
Customer:
I want a refund for an App Store purchase
```

Retrieved historical issue:

``` text
Just got conned into an app store purchase which didnt give me
an option to say no. How do I get a refund?
```

Retrieved support guidance:

``` text
We'd like to get you pointed in the right direction.
You'll want to contact our iTunes Advisors...
```

------------------------------------------------------------------------

# 9. Evaluation Harness

The project includes:

``` text
src/evaluate_retriever.py
src/analyze_errors.py
```

The current automated harness evaluates the 200-example set and reports
intent-routing accuracy.

Current baseline result:

``` text
Total examples: 200
Correct: 142
Incorrect: 58
Intent accuracy: 71.00%
```

The error-analysis script groups mistakes by expected and predicted
intent.

## Important distinction

Intent accuracy is **not the same as reply quality**.

A reply can be useful even if the internal intent label is imperfect.

Conversely, the correct intent does not guarantee that the retrieved
reply is relevant.

Therefore the final evaluation should measure both:

``` text
Intent routing
+
Reply relevance / groundedness
```

------------------------------------------------------------------------

# 10. LLM-as-Judge Evaluation

A complete reply-quality evaluation should use an LLM judge in addition
to automated metrics.

The proposed judge rubric is:

  Criterion                              Score
  ------------------------------------ -------
  Issue relevance                         0--2
  Groundedness in retrieved evidence      0--2
  Helpfulness                             0--2
  Professional tone                       0--2
  Hallucination / unsupported claims      0--2

Total:

``` text
10 points
```

## Judge prompt

A judge can be given:

``` text
Customer query:
{query}

Retrieved historical customer issue:
{retrieved_issue}

Retrieved AppleSupport response:
{retrieved_response}

Evaluate the proposed reply.

Score:
1. Relevance: 0-2
2. Groundedness: 0-2
3. Helpfulness: 0-2
4. Professional tone: 0-2
5. Hallucination/unsupported claims: 0-2

Return:
{
  "score": 0-10,
  "reason": "...",
  "hallucination": true/false
}
```

## Human agreement

For a proper final benchmark, a sample of the same replies should also
be rated by a human using the identical rubric.

Agreement can then be reported using:

-   Pearson/Spearman correlation for total scores.
-   Mean absolute error between human and LLM scores.
-   Agreement rate on acceptable/unacceptable replies.

### Current status

The current repository contains the automated intent evaluation, but
**human-vs-LLM judge agreement has not yet been completed**.

It should not be claimed as completed until those ratings are actually
collected.

------------------------------------------------------------------------

# 11. Baselines

The system should be compared against at least two simpler baselines.

## Baseline 1 --- Trivial baseline

Always predict the most frequent intent.

For response generation, return the most common AppleSupport response
from the training data.

This establishes a lower bound.

## Baseline 2 --- Simple keyword baseline

Use only keyword matching:

``` text
battery -> battery_charging
wifi -> wifi_connectivity
icloud -> icloud_apple_id
refund -> app_store_purchases
macbook -> mac_macos
```

Then retrieve the first matching historical response.

## Proposed system

The proposed system combines:

``` text
Intent routing
+
TF-IDF similarity
+
Cosine similarity
+
Domain keyword reranking
```

### Results table

The final report should contain measured results rather than invented
numbers:

  -----------------------------------------------------------------------------
  System                  Intent Accuracy        Reply Quality Notes
  ------------------ -------------------- -------------------- ----------------
  Majority/trivial                Measure              Measure Most frequent
  baseline                                                     class

  Keyword baseline                Measure              Measure Keyword-only

  Proposed TF-IDF +      **71.00% current              Measure Current
  reranking             intent baseline**                      implementation
  -----------------------------------------------------------------------------

Do not fill missing metrics with guessed values.

------------------------------------------------------------------------

# 12. Failure Analysis

The current evaluation produced several recurring failure modes.

## Failure Mode 1 --- iOS update vs battery

Example:

``` text
iOS11 has destroyed my new iPhone SE. Eats battery...
```

Expected:

``` text
software_ios
```

Predicted:

``` text
battery_charging
```

### Hypothesis

The word `battery` is a highly strong signal and can override the
broader software-update context.

### Improvement

Use phrase-level context and give update-related phrases more weight
when the query explicitly attributes the battery problem to an iOS
update.

------------------------------------------------------------------------

## Failure Mode 2 --- Notifications vs software update

Example:

``` text
Messages app starts showing message notifications after upgrade to iOS 11...
```

Expected:

``` text
software_ios
```

Predicted:

``` text
notifications_messages
```

### Hypothesis

The query contains multiple valid topics.

### Improvement

Allow multi-intent scoring and select the intent associated with the
customer's main complaint rather than the first matching keyword.

------------------------------------------------------------------------

## Failure Mode 3 --- Generic messages become unknown

Example:

``` text
Yes, version 11.1.1
```

Expected:

``` text
software_ios
```

Predicted:

``` text
unknown
```

### Hypothesis

Very short replies contain too little lexical information.

### Improvement

Use conversation history instead of classifying isolated tweets.

------------------------------------------------------------------------

## Failure Mode 4 --- Account/security overlap

Example:

``` text
scammers steal Apple ID
```

Expected:

``` text
security_fraud
```

### Hypothesis

Terms such as `Apple ID`, `password`, and `account` strongly overlap
with the iCloud/account intent.

### Improvement

Give security/fraud indicators priority over generic account terms.

------------------------------------------------------------------------

## Failure Mode 5 --- App Store refund vs generic refund

Example:

``` text
I want a refund for an App Store purchase
```

A pure TF-IDF retrieval can initially return:

``` text
I want a refund for my laptop
```

### Hypothesis

`refund` has strong lexical weight but does not identify the
product/domain.

### Improvement

Use domain-aware reranking so that `App Store`, `purchase`, and `refund`
jointly outweigh generic `refund` similarity.

------------------------------------------------------------------------

# 13. "What is misleading about my headline number?"

The headline number is currently:

``` text
71.00% intent accuracy
```

This number is useful, but it is **not a complete measure of system
quality**.

There are several reasons:

1.  The golden set was initially keyword-assisted rather than fully
    human-labelled.
2.  The metric measures intent routing, not whether the final support
    reply is helpful.
3.  Some examples contain multiple legitimate topics, making a
    single-label taxonomy imperfect.
4.  A wrong intent can still retrieve a relevant response.
5.  A correct intent can still retrieve an irrelevant historical
    response.
6.  The evaluation set is relatively small compared with the full
    dataset.
7.  Social-media messages can be extremely short and ambiguous.

Therefore:

> 71% should be interpreted as a baseline intent-routing measurement,
> not as "the system gives correct support replies 71% of the time."

A stronger final headline should report separate metrics for:

``` text
Intent accuracy
Retrieval relevance
Groundedness
Human/LLM reply-quality agreement
```

------------------------------------------------------------------------

# 14. Installation

## Requirements

Recommended:

-   Windows
-   Python 3.10+
-   PowerShell
-   Git

The project was developed and tested locally with Python 3.14.

------------------------------------------------------------------------

# 15. Complete Setup From Scratch

## Step 1 --- Clone the repository

``` powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Then:

``` powershell
cd hiver-sde-assignment
```

------------------------------------------------------------------------

## Step 2 --- Create a virtual environment

``` powershell
python -m venv venv
```

This creates:

``` text
venv/
```

inside the project.

------------------------------------------------------------------------

## Step 3 --- Activate the virtual environment

PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

You should see something similar to:

``` text
(venv) PS C:\...\hiver-sde-assignment>
```

If PowerShell blocks activation:

``` powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run:

``` powershell
.\venv\Scripts\Activate.ps1
```

------------------------------------------------------------------------

## Step 4 --- Install dependencies

``` powershell
pip install -r requirements.txt
```

If FastAPI/Uvicorn are missing:

``` powershell
pip install fastapi uvicorn
```

------------------------------------------------------------------------

# 16. Dataset Setup

Download:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Place:

``` text
twcs.csv
```

inside:

``` text
data/
```

Final path:

``` text
data/twcs.csv
```

------------------------------------------------------------------------

# 17. Build Everything From Scratch

Run these commands from the project root.

### Extract AppleSupport

``` powershell
python src/extract_applesupport.py
```

### Extract conversations

``` powershell
python src/extract_applesupport_conversations.py
```

### Create support pairs

``` powershell
python src/create_support_pairs.py
```

### Analyze topics

``` powershell
python src/topic_analysis.py
```

### Create golden set

``` powershell
python src/create_golden_set.py
```

### Build retriever

``` powershell
python src/build_retriever.py
```

### Train baseline intent classifier

``` powershell
python src/train_intent_classifier.py
```

### Evaluate

``` powershell
python src/evaluate_retriever.py
```

### Analyze errors

``` powershell
python src/analyze_errors.py
```

------------------------------------------------------------------------

# 18. Run the Model Directly

From the project root:

``` powershell
python src/generate_response.py
```

Enter a query.

## Query 1

``` text
My iPhone battery is draining very quickly
```

Expected intent:

``` text
battery_charging
```

## Query 2

``` text
My WiFi keeps disconnecting on my iPhone
```

Expected intent:

``` text
wifi_connectivity
```

## Query 3

``` text
I can't log into my iCloud account
```

Expected intent:

``` text
icloud_apple_id
```

## Query 4

``` text
I want a refund for an App Store purchase
```

Expected intent:

``` text
app_store_purchases
```

## Query 5

``` text
My MacBook is running extremely slowly
```

Expected intent:

``` text
mac_macos
```

## Query 6

``` text
My iPhone isn't showing notifications for new messages
```

Expected intent:

``` text
notifications_messages
```

## Query 7

``` text
My iPhone screen is completely black
```

Expected intent:

``` text
hardware_device
```

## Query 8

``` text
I received a suspicious email asking for my Apple ID password
```

Expected intent:

``` text
security_fraud
```

------------------------------------------------------------------------

# 19. Run the FastAPI Server

If the environment is activated:

``` powershell
python -m uvicorn src.api:app --reload
```

You should see:

``` text
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open.

------------------------------------------------------------------------

# 20. Test the API

Open:

``` text
http://127.0.0.1:8000
```

Expected:

``` json
{
  "message": "AppleSupport Customer Support API",
  "status": "running"
}
```

------------------------------------------------------------------------

# 21. Swagger UI

Open:

``` text
http://127.0.0.1:8000/docs
```

Find:

``` text
POST /reply
```

Click:

``` text
Try it out
```

Enter:

``` json
{
  "query": "I want a refund for an App Store purchase"
}
```

Click:

``` text
Execute
```

------------------------------------------------------------------------

# 22. Example API Response

A typical response looks like:

``` json
{
  "query": "I want a refund for an App Store purchase",
  "intent": "app_store_purchases",
  "response": "We'd like to get you pointed in the right direction. You'll want to contact our iTunes Advisors: https://t.co/SDIe7UiyJN",
  "similarity": 0.3536,
  "retrieved_issue": "Just got conned into an app store purchase which didnt give me an option to say no. How do I get a refund?"
}
```

The exact response and score can vary because retrieval depends on the
indexed historical examples.

------------------------------------------------------------------------

# 23. Project Structure

``` text
hiver-sde-assignment/
│
├── data/
│   ├── twcs.csv
│   ├── applesupport_raw.csv
│   ├── applesupport_conversations.csv
│   ├── applesupport_pairs.csv
│   ├── golden_set.csv
│   └── retriever_evaluation.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── retriever.pkl
│   ├── retrieval_data.pkl
│   ├── intent_vectorizer.pkl
│   └── intent_classifier.pkl
│
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
│
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# 24. Technologies

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
-   PowerShell
-   Git

------------------------------------------------------------------------

# 25. What I Would Do With One More Week

If given another week, I would prioritize:

### Day 1--2: Better semantic retrieval

Replace TF-IDF with sentence embeddings and compare:

``` text
TF-IDF
vs
Sentence Transformers
```

This should improve paraphrase matching.

### Day 3: Cross-encoder reranking

Use a cross-encoder on the top 20 retrieved cases to improve relevance.

### Day 4: Human evaluation

Manually label:

-   200+ golden examples
-   50--100 generated replies

Collect human quality scores using the same judge rubric.

### Day 5: LLM grounded generation

Use an LLM only after retrieval and pass the retrieved evidence into the
prompt.

The model would be explicitly instructed:

``` text
Do not introduce information that is not supported by the retrieved evidence.
```

### Day 6: Evaluation

Add:

-   Precision@K
-   Recall@K
-   MRR
-   Reply-quality score
-   Groundedness score
-   Hallucination rate
-   Human/LLM agreement

### Day 7: Production polish

Add:

-   Docker
-   Better API validation
-   Logging
-   Confidence thresholds
-   Monitoring
-   A simple support-agent frontend

------------------------------------------------------------------------

# 26. Limitations

-   The current intent taxonomy is manually designed.
-   The current golden set requires final human validation before being
    described as fully hand-labelled.
-   TF-IDF is weaker than embedding-based retrieval for paraphrased
    queries.
-   Some Twitter messages are extremely short or ambiguous.
-   Some historical AppleSupport responses are generic.
-   Multiple issues can occur in one customer message.
-   Historical responses may contain Twitter-specific language and
    links.
-   The current 71% number measures intent routing, not end-to-end reply
    quality.
-   Human-vs-LLM judge agreement has not yet been established in the
    current implementation.

------------------------------------------------------------------------

# 27. Final Quick Start

After cloning the repository:

``` powershell
cd hiver-sde-assignment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Make sure:

``` text
data/twcs.csv
```

exists.

If the processed data/models are already included:

``` powershell
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

with:

``` text
POST /reply
```

------------------------------------------------------------------------

# 28. Submission Checklist

Before submitting the assignment:

-   [ ] `README.md` is complete.
-   [ ] `requirements.txt` is present.
-   [ ] `.gitignore` excludes `venv/` and large raw data.
-   [ ] API starts successfully.
-   [ ] `/docs` works.
-   [ ] `/reply` returns HTTP 200.
-   [ ] At least 150--250 golden examples exist.
-   [ ] Golden examples are manually reviewed.
-   [ ] Evaluation script runs from a clean environment.
-   [ ] Trivial baseline is measured.
-   [ ] Keyword baseline is measured.
-   [ ] Proposed system is measured.
-   [ ] Reply-quality evaluation is run.
-   [ ] LLM judge is compared against human ratings.
-   [ ] Failure analysis contains real examples.
-   [ ] Headline metric is clearly qualified.
-   [ ] No fabricated evaluation numbers are included.

------------------------------------------------------------------------
