# Part 4: AI Solution Design for a Business Problem

**Domain:** Customer Support  
**Business Problem:** Automated Ticket Sentiment Routing  
**AI Task:** Text Classification (Sentiment Analysis)  
**Model:** DistilBERT (fine-tuned Transformer)

---

## What is the problem?

Customer support teams get a lot of tickets every day. Right now, human agents have to manually read each ticket, figure out how urgent or frustrated the customer is, and then send it to the right team. This takes a lot of time and effort.

Some quick numbers to understand the problem:
- About **2,778 tickets come in every month**
- Agents spend **454 hours per month** just on sorting and routing tickets
- Around **7% of tickets get sent to the wrong team** (misrouting)
- Average time to resolve a ticket is **30.3 hours**

The idea is to use an AI model to automatically read the ticket, detect the customer's sentiment (positive, neutral, or negative), and route it to the right team — without any manual work.

## My Solution

I used a **DistilBERT model** (a smaller and faster version of BERT) and fine-tuned it on customer support tickets. The model classifies each ticket into one of three categories:
- `positive` — happy or satisfied customer
- `neutral` — just asking for info or checking status
- `negative` — frustrated or angry customer

After classification, a simple routing rule sends the ticket to the correct queue automatically.

## What improvement do we expect?

| KPI | Before AI | After AI (Target) |
|-----|-----------|-----------------|
| Avg resolution time | 30.3 hours | ≤ 20 hours |
| Manual triage hours/month | 454 hours | ≤ 100 hours |
| Misrouting error rate | 6.96% | ≤ 2.0% |
| Customer Satisfaction (CSAT) | 7.04 / 10 | ≥ 8.0 / 10 |

## Folder Structure

```
part-4-ai-solution-design/
├── README.md
├── solution_report.md
└── diagrams/
    └── solution_architecture.png
```

## Files in this repo

| File | What it contains |
|------|-------------|
| `solution_report.md` | Full detailed report covering all 8 tasks |
| `diagrams/solution_architecture.png` | Diagram showing how the whole system works end to end |

## Data used for reference

- `ai_usecase_reference_catalog.csv` — helped me pick the right domain and model (Row 9: Customer Support)
- `business_kpi_sample.csv` — gave me the baseline numbers like average tickets per month and resolution time
