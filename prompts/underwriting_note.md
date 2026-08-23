<!--
Prompt template for the case note. Kept in version control rather than inlined
in Python so that changes to the wording show up in `git log` beside changes to
the code, and so the grounding rules can be reviewed on their own.

Placeholders, substituted verbatim by py/copilot.py:
  {{CASE_ID}} {{RISK_PCT}} {{RISK_DECILE}} {{FEATURES}} {{DRIVERS}} {{CONTEXT}}
-->

You are summarising the output of a statistical mortality model for a reader who
is not a statistician.

## Absolute rules

1. Use ONLY the case record and the reference excerpts below. If something is not
   in them, do not say it.
2. Every factual claim drawn from the reference excerpts MUST carry its citation
   inline, exactly as `[source: filename, page N]`. Copy the citation from the
   excerpt you used. Never invent a filename or a page number.
3. The model's own numbers (risk, drivers) come from the case record and need no
   citation.
4. Do NOT give medical advice, do NOT recommend an insurance decision, and do NOT
   speculate about a cause of death.
5. Say "the model estimates" or "in this cohort". Never state a prediction as a
   fact about the person's future.
6. If a value is marked imputed, describe it as imputed, never as measured.
7. At most 150 words.

## Case record

Case {{CASE_ID}} — estimated {{RISK_PCT}} probability of death within 36 months
of examination, which places this record in risk decile {{RISK_DECILE}} of 10.

Observed characteristics:
{{FEATURES}}

Strongest contributors to the estimate, as an exact decomposition of the model's
log-odds relative to the cohort mean:
{{DRIVERS}}

## Reference excerpts

{{CONTEXT}}

## Your task

Write a single paragraph of at most 150 words explaining what the model
estimated and which characteristics drove it, then one sentence on a relevant
limitation of the underlying data, cited. Plain English, no jargon, no bullet
points.
