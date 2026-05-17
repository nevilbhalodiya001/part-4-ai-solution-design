# Part 4: AI Solution Design for a Business Problem
# Solution Report — Customer Support Ticket Sentiment Routing

---

## Task 1: Business Domain Selected

**Domain:** Customer Support  
**Reference:** AI Use Case Reference Catalog — Row 9: *Ticket sentiment routing*

I chose Customer Support because it's a very common real-world problem where AI can make a big difference. Companies deal with thousands of support tickets, and automating the routing process can save a lot of time and money.

---

## Task 2: Define the Business Problem

### What problem is being solved?

A customer support team receives around **2,778 tickets every month** through email and chat. Every ticket has to be manually read by an agent, the sentiment is guessed (is the customer angry? just asking a question? happy?), and then it gets sent to the right team.

This whole triage process is done by humans right now, and it has some big problems:
- It takes an average of **30.3 hours** to fully resolve a ticket
- Agents spend **453.8 hours per month** just on triage — not even solving problems
- About **6.96% of tickets get misrouted** (wrong team gets the ticket)
- Customer Satisfaction Score is only **7.0 out of 10**

The goal is to automate this triage step using an AI model so agents can focus on actually solving problems instead of sorting tickets.

### Who are the users and stakeholders?

| Stakeholder | Their Role |
|-------------|------|
| Support agents | They receive the pre-routed tickets and respond to customers |
| Tier-2 escalation team | They handle the most urgent or angry customer tickets |
| Operations manager | Keeps track of SLA and customer satisfaction |
| Customer | The person who actually benefits from faster, more accurate support |
| CTO / AI team | Responsible for building and maintaining the AI model |

### How does the current process work?

1. A customer sends a support ticket (email or chat).
2. A human triage agent reads the message and tries to figure out how the customer is feeling.
3. The agent manually assigns a priority and routes the ticket to the right team (urgent, standard, or FAQ).
4. Another agent actually solves the issue.

As ticket volume keeps growing (it went from 2,466 to 3,154 per month in the data), this manual process becomes a bigger and bigger bottleneck.

### What's wrong with the current process?

- **Inconsistent:** Different agents might classify the same ticket differently.
- **Slow:** By the time triage is done, the customer has already been waiting for hours.
- **Expensive:** Over 450 agent-hours every month are spent just on routing, not resolving.
- **Error-prone:** Around 7% of tickets end up with the wrong team.
- **No 24/7 coverage:** Manual triage doesn't happen at night or on weekends unless someone works overtime.

---

## Task 3: Identify the AI Task Type

**AI Task Type: Text Classification (Sentiment Classification)**

### Why text classification?

Each ticket is a piece of text, and we want to label it as one of three categories: positive, neutral, or negative. This is a classic multi-class text classification problem.

We already have historical tickets with labels (assigned by agents), so we can use this as training data for a supervised ML model.

### How does the system work step by step?

The solution has two parts:

1. **Sentiment classification** — The AI model reads the ticket text and predicts whether the sentiment is positive, neutral, or negative.
2. **Rule-based routing** — A simple if-else rule maps the predicted sentiment to the correct support queue (for example: negative + urgent keywords → escalate immediately).

This two-step design is clean — the ML model handles the hard part (understanding text), and the routing rules are easy to explain to stakeholders.

---

## Task 4: Data Requirement Plan

### What kind of data do we need?

We need **text data** (the actual customer messages) along with some **metadata** like the channel it came from and whether the ticket was flagged as urgent.

### Where does the data come from?

| Source | What we get from it |
|--------|-------------|
| Support ticket CRM (like Zendesk or Freshdesk) | Historical ticket text + labels assigned by agents |
| Email archive | Full message body and subject line |
| Chat logs | Full chat transcript |
| KPI database | Resolution times, CSAT scores, escalation history |

### What features does the model use?

| Feature | Data Type | Example |
|---------|------|---------|
| `customer_message` | Raw text | "My refund is still pending and this is frustrating" |
| `channel` | Category | email, chat, phone, social, app |
| `word_count` | Number | 12 |
| `urgent_flag` | 0 or 1 | 1 = urgent |
| `hour_of_day` | Number | 14 (meaning 2 PM) |

### What is the target label?

`sentiment_label` — three possible values:
- `negative` — customer is frustrated, angry, or complaining
- `neutral` — customer is asking for info or checking status
- `positive` — customer is happy or satisfied

### How do we collect and label the data?

- **Historical labels:** Export 12–18 months of past tickets and have agents manually label a random sample of ~5,000 tickets.
- **Active learning:** Start with a weak model, find the tickets it's least confident about, and send those for human review. Add them to the training set to improve the model over time.
- **Continuous pipeline:** New tickets flow through a preprocessing service and get added to the dataset for retraining every quarter.

### Possible data quality issues

| Risk | How we handle it |
|------|------------|
| Most tickets are neutral (class imbalance) | Use stratified sampling and class-weight adjustment |
| Different agents label the same ticket differently | Measure inter-annotator agreement (Cohen's Kappa ≥ 0.75 required) |
| Slang or non-English text | Detect language first; keep English only for v1 |
| PII in messages (names, account numbers) | Redact PII using regex and NER before training |
| Very short messages like "please help" | Flag as low-confidence and send to human review |

---

## Task 5: Model Recommendation

### Recommended Model: Fine-tuned DistilBERT

I recommend using **DistilBERT fine-tuned on customer support tickets**.

DistilBERT is basically a lighter version of BERT — it's 40% smaller and 60% faster, but keeps 97% of BERT's performance. Here's why I picked it:

1. **It understands context.** Unlike older models, it knows that "not great" and "great" mean very different things.
2. **Transfer learning works well here.** DistilBERT is already pre-trained on huge amounts of text, so fine-tuning it on ~5,000 support tickets is enough.
3. **It's fast enough for real-time use.** It can give a prediction in under 50ms per ticket.
4. **The `[CLS]` token is perfect for classification.** BERT-style models have a special token that captures the meaning of the whole sentence — ideal for sentiment classification.

### How does it compare to other options?

| Model | Good things | Bad things | My verdict |
|-------|------|------|---------|
| Logistic Regression + TF-IDF | Fast, easy to explain | Misses context and negations | Only as a baseline |
| LSTM | Understands word order | Slower to train, needs more data | Good fallback |
| BERT-base | Best accuracy | Very large (110M params), slower | Use if speed doesn't matter |
| **DistilBERT** | **Fast + accurate + smaller size** | **Tiny accuracy drop vs BERT** | **My recommendation** |
| GPT-style LLM | Very powerful | Overkill for this task, expensive | Not needed here |

### How does the architecture look?

```
Customer Ticket Text
        |
  [Preprocessing]
  Lowercase, strip PII, truncate to 512 tokens
        |
  [DistilBERT Encoder]
  Tokenizer → 6-layer Transformer → [CLS] vector (768-dim)
        |
  [Classification Head]
  Dense(256, ReLU) → Dropout(0.2) → Dense(3, Softmax)
        |
  [Sentiment Output]
  Example: {negative: 0.81, neutral: 0.13, positive: 0.06}
        |
  [Business Routing Rules]
  negative + urgent_flag=1 → Priority Escalation Queue
  negative only            → Standard Escalation Queue
  neutral                  → Standard Support Queue
  positive                 → Self-service / Close suggestion
```

---

## Task 6: Evaluation Plan

### Technical metrics (how do we know the model is performing well?)

| Metric | What it means | Target |
|--------|------------|--------|
| **F1-score (macro)** | Overall balance of precision and recall across all 3 classes | ≥ 0.88 |
| **Precision** | Of all tickets we escalated, how many actually needed it | ≥ 0.90 |
| **Recall (negative class)** | Of all truly angry tickets, how many did we catch | ≥ 0.92 |
| **AUC-ROC (one-vs-rest)** | How well the model separates each class | ≥ 0.95 |
| **Inference latency** | How fast does the model predict | < 100ms |

The most important metric here is **recall on the negative class**. Missing an angry customer and routing them incorrectly is much worse than occasionally escalating a neutral ticket.

### Business metrics (does it actually help?)

| KPI | Current | AI Target |
|-----|-----------------|-----------|
| Average resolution time | 30.3 hours | ≤ 20 hours (−34%) |
| Manual triage hours/month | 453.8 hours | ≤ 100 hours (−78%) |
| Misrouting error rate | 6.96% | ≤ 2.0% |
| Customer Satisfaction (CSAT) | 7.04 / 10 | ≥ 8.0 / 10 |

### What can go wrong?

| Failure Type | Description | How to handle it |
|---|---|---|
| Angry customer gets low priority | Model misses negative sentiment | High recall threshold; add urgent keyword override |
| Confident wrong prediction | Model is 95% sure a sarcastic complaint is positive | Augment training data with sarcasm; human review for edge cases |
| Model drift | Language patterns change over time | Monthly monitoring; retrain every quarter |
| Ambiguous short messages | "Please help." has no clear sentiment | If confidence < 0.6, send to human triage |

### How do we validate before going live?

1. **Shadow mode (Month 1–2):** Run the model quietly in the background. Don't act on its predictions — just compare them to what humans decided.
2. **Partial rollout (Month 3):** Let the model handle 50% of tickets on lower-risk channels like chat.
3. **Full rollout (Month 4+):** Model handles all tickets. Anything below 0.60 confidence goes to human review.
4. **Ongoing:** Track F1-score weekly. Alert the team if it drops more than 3% from baseline.

---

## Task 7: Responsible AI Considerations

### 1. Bias in training data

If different agents labeled tickets differently, or if some channels (like non-native English speakers) were labeled poorly, the model will learn those biases. This could mean worse routing quality for certain customer groups.

**Fix:** Audit the training data by channel and agent. Check if F1-score is consistent across different groups.

### 2. Wrong predictions

If an angry customer gets misclassified as neutral, they end up in the wrong queue and have to wait longer — which makes things worse.

**Fix:** Keep human review for low-confidence predictions. Add keyword override rules for words like "urgent", "lawsuit", "escalate", "unacceptable".

### 3. Privacy / PII in ticket text

Customers often include personal information in their messages — names, email addresses, account numbers, sometimes even health details. We can't use this raw text for training without proper handling.

**Fix:** Run PII redaction (regex + NER) before any text enters the training pipeline. Encrypt stored data. Delete training examples after 24 months.

### 4. Agents over-relying on AI

If agents blindly trust the model and stop reading tickets carefully, they'll miss things the model gets wrong.

**Fix:** Train agents that the model is a helper, not the decision-maker. Show the confidence score alongside the suggestion. Let agents override and log corrections — those corrections become new training data.

### 5. Impact on support agents

Automating triage might make agents feel like their jobs are at risk or that their skills don't matter anymore.

**Fix:** Be transparent about what the AI does and doesn't do. Frame it as removing boring, repetitive work so agents can focus on actually helping customers. Involve agents in the testing process so they feel ownership.

### 6. Need for ongoing human oversight

Without someone actively monitoring the model, performance can silently get worse over time as customer language changes.

**Fix:** Assign a model owner who sends weekly performance reports. If macro-F1 drops below 0.80, revert to manual triage until retraining is complete.

---

## Task 8: Final Solution Summary

| Section | Summary |
|---------|--------|
| **Problem** | Customer support tickets are manually triaged and routed — slow (30.3 hrs avg), expensive (454 hrs/month), and error-prone (7% misrouting) |
| **AI Solution** | A text classification model that reads each ticket, predicts sentiment (positive/neutral/negative), and automatically routes it to the correct team |
| **AI Task Type** | Multi-class text classification (sentiment analysis) |
| **Data Needed** | Historical labeled support tickets (text + sentiment label), channel type, urgent flag, timestamp |
| **Recommended Model** | DistilBERT fine-tuned on customer support tickets, with a Dense(3, softmax) classification head |
| **Expected Business Impact** | Resolution time: 30.3 → ≤ 20 hrs · Triage hours: 454 → ≤ 100/month · Error rate: 7% → ≤ 2% · CSAT: 7.0 → ≥ 8.0 |
| **Key Risks** | Biased training data, PII in text, agents over-relying on AI, model drift over time |
| **Mitigation Plan** | PII redaction, confidence-based human fallback, keyword override rules, monthly monitoring, quarterly retraining |
| **Deployment Plan** | Shadow mode (Month 1–2) → 50% rollout (Month 3) → Full rollout with monitoring (Month 4+) |

---

*Submitted as Part 4 of AI Solution Design — Masai School Data Science Programme*
