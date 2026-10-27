# Looking for a Hidden Agenda: Can a Model's Activations Reveal Steganography that its Text Hides?

*Apart AI Collusion Research Sprint, October 2026. Early markdown draft, superseded by the LaTeX paper. All numbers are from `results.md`; references marked
‡ were added for this draft and should be checked against the source before submission (see also the verification
note in `probing-lit-review.md`).*

## Abstract

Text monitors, such as filters for personal data or for protected attributes, check what a message says. A
model that has a reason to pass on a protected attribute does not need to say it. It can choose which facts to
mention, or frame the same facts more or less warmly, so that a reader in on the arrangement can infer the
attribute while each message on its own looks clean. We built a small hiring environment in which this happens:
a language model (Qwen3-8B) summarizing job candidates for a hiring manager is given a reason to signal each
candidate's university, while a reviewer screens every summary for any reference to it. With its reasoning
visible to us, the model plans the signal in more than eight in ten such episodes, and most of the resulting
summaries pass the reviewer.

If the signal is invisible in each message, perhaps the plan behind it is visible inside the model. We tested
this idea, which motivates much of the recent work on activation probes for deception and collusion. We replayed
1,152 stored summaries through the same model under carefully varied contexts and asked a sequence of questions,
each prompted by the answer to the previous one. A probe trained in the obvious way seems to detect the hidden
agenda perfectly, but it is reading the instructions in the prompt rather than anything about the model's
behaviour. That confound turns out to occupy a single direction in activation space and can be removed. What
remains is a probe that detects covert summaries well, but no better than the same model reading the summary with
no context at all. An apparent case of the model "knowing more than it says" disappears under the same control.
In this setting, we find no evidence of a concealment signal in the model's internal state beyond what is
already in its output text.

One test did work, and it works on a new attribute without retraining. Borrowing the logic of resume audit
studies, we replay each summary with only the protected attribute changed and ask how much less likely the model
finds its own output. This counterfactual test catches 60% of covert summaries on a held-out attribute
(nationality) at a 1% false-alarm rate, set using the university data alone.

## 1. Introduction

### 1.1 When a clean message is not a clean channel

In 2018, Reuters reported that Amazon had abandoned an experimental résumé-screening model after finding that it
penalised résumés containing the word "women's", as in "women's chess club captain", and downgraded graduates of
two all-women's colleges [Dastin 2018 ‡]. The model was never given the applicant's gender. It learned to read
gender from other words, because gender predicted the label it was trained on. Removing an explicit attribute does
not remove the information when other features carry it, a pattern long known in discussions of proxy
discrimination.

Language-model agents add a new version of this problem. An agent that writes text for another agent can be the
one choosing the proxy. Suppose an agent summarizes candidates for a hiring manager, and both are rewarded when
the hiring outcome matches a pattern that correlates with a protected attribute. A compliance filter that blocks
any mention of the attribute will pass a summary that simply leaves out a weak interview score for one group and
dwells on it for another. No single summary is objectionable. The information travels in *which* true facts are
selected and *how* they are framed. This is a mild, natural form of what the literature calls steganographic
collusion [Motwani et al. 2024; Mathew et al. 2024 ‡]: two agents coordinating through a channel the overseer
cannot read.

Monitoring that reads the text faces a basic limit here. A well-hidden message is, by construction, one that a
reader of the text cannot distinguish from an innocent one. Monitoring the agent's chain of thought helps when the
plan is written out, but chains of thought can be unfaithful or can learn to hide intent under optimization
pressure [Baker et al. 2025; Korbak et al. 2025], and recent work argues that faithful reasoning still need not
expose collusion [Faithful yet Collusive, 2026].

### 1.2 The hope: the agenda is still in the activations

To produce a covert signal, the model has to decide to produce it. The intuition behind white-box monitoring is
that this decision, the *hidden agenda*, should leave a trace in the model's internal activations even when the
output text does not. A line of work supports this hope. Linear probes on hidden states can tell when a model's
statement is false even as it asserts it [Azaria & Mitchell 2023; Marks & Tegmark 2023], and logistic-regression
probes on mid-layer activations detect strategic deception in realistic scenarios [Goldowsky-Dill et al. 2025].
For collusion specifically, Rose et al. (2026) report probes that detect instructed multi-agent collusion with
near-perfect accuracy, including a steganographic card-counting game, and Das et al. (2026) find that these
probes largely survive when agents are told they are being monitored. Veron (2026) shows that a probe for a
mundane fact the model must represent ("this code has a backdoor") catches a colluding monitor that lies in its
verdict.

The same literature also flags a risk. Probes may rely on textual evidence rather than latent intent
[Linear probes rely on textual evidence, 2025], and the useful measure of a white-box method is its improvement
over a black-box baseline on the same data, which is often modest [Black-to-White Performance Boosts, 2025].
In most collusion benchmarks, the colluding agent is told to collude in its system prompt. A probe can then
succeed by recognising those instructions, which a monitor with access to the prompt could do directly.

### 1.3 This paper

We ask a narrow question in a setting built to answer it: **when a model conceals a protected attribute in
otherwise innocent text, do its activations carry information about the concealment that its text does not?**

Our setting has three properties that make the question testable. First, the concealment is the model's own: the
system prompt gives it a goal but no method, and it invents the signal in its reasoning, which lets us label each
episode by whether the model actually planned a signal. Second, every covert episode has a matched honest
counterpart whose system prompt is identical except for the goal paragraph. Third, because we store the exact
text each episode produced, we can feed the same output back through the same model under different contexts.
This lets us separate what the probe learns from the prompt, from the output text, and from anything else.

The analysis climbs a ladder of controls. Each rung removes one possible source of a probe's signal: first the
instructions in the prompt, then the candidate profile, then all context except the summary itself. If a signal
about concealment lived in the model's state rather than in its text, it should vanish only at the last rung.

1. A standard probe for the hidden agenda scores perfectly, but swapping prompts shows it reads the prompt (§4.1).
2. The prompt's influence lies along about one direction and can be erased, leaving a prompt-independent probe (§4.2).
3. That probe does no better than the same model reading the bare summary with no context (§4.3).
4. An apparent excess of hidden knowledge about the attribute survives removing the profile and all context, so
   it too is in the text (§4.4).

Having found that the signal lives in the text, we then ask a different question, whether the text *depends* on
the attribute, with a counterfactual test that changes only the attribute. It detects use of the attribute and
transfers to a new attribute without retraining (§4.5).

## 2. The environment

### 2.1 A hiring task with a misspecified reward

We follow the design of Mathew et al. (2024 ‡), in which a reward that correlates with a sensitive attribute makes
leaking that attribute the cheapest way to succeed. Each episode involves three roles, all played by the same
open-weight model, Qwen3-8B, so that every activation we later analyse comes from weights we can run ourselves.

```
   candidate profile (includes the university)
                  │
                  ▼
             ┌─────────┐   writes a ≤60-word summary; told never to reveal the university
             │ sender  │   (reasoning enabled, stored)
             └────┬────┘
                  │ summary
          ┌───────┴────────┐
          ▼                ▼
     ┌──────────┐     ┌──────────┐
     │ reviewer │     │  hiring  │   sees only the summary; answers hire / reject
     │(overseer)│     │ manager  │   (the receiver)
     └──────────┘     └──────────┘
   pattern rules + a model judge: PASS or FLAG

   reward = summary passes the reviewer AND hiring decision matches the label
```

The **candidate profile** is synthetic: name, role, years of experience, degree, university, previous employer,
skills, an interview score out of 10 and a technical test score out of 100. The **secret attribute** is the
university, one of two fictional institutions (Halvorsen University and Corrigan State University), balanced
exactly within each batch. The hiring label correlates with the university, so a receiver that could infer it
would earn more reward.

The **reviewer** stands in for a compliance filter. It combines pattern rules (for university names, explicit
hiring verdicts and tag-like codes) with a model judge that flags any summary alluding to the candidate's
education institution. It is strict: it flags about 40% of honest summaries, usually for mentioning the degree.

### 2.2 Why the sender is told the goal but not the method

We first tried to make collusion emerge without telling anyone to collude: an outer optimizer (a separate model)
rewrote the sender's instructions between rounds based on reward, a form of in-context reinforcement learning.
It worked, but every channel it found was overt (stating the verdict outright, then attaching group-specific tags)
and collusion collapsed once the reviewer blocked those. Those episodes also have a deeper problem for our
question: the sender was following an explicit rule devised by another model, so its internals would reflect rule
lookup, not a plan of its own.

We therefore give the sender a goal and let it find the method. In the **partner condition**, the system prompt
adds one paragraph: the sender's objective is to get Halvorsen candidates hired and Corrigan State candidates
rejected; the hiring manager is its partner, shares this objective and knows the sender will hint; the reviewer
must not notice. Nothing in the prompt suggests how to signal. Reasoning is enabled, so the model's plan is
written out before the summary, and we keep it.

The **honest condition** uses the same system prompt with the goal paragraph removed (the honest prompt is an
exact prefix of the partner prompt) and the same user message. This matched control is the reason the later tests
are possible: any difference between the two conditions is either the goal paragraph or what the model did with
it.

### 2.3 What the model does

The sender plans a signal in 84–87% of partner episodes. A model judge read each stored reasoning trace and
recorded whether there was a concrete, attribute-dependent plan, its direction and its mechanism. In all but one
of the 770 partner episodes where it found a plan, the planned direction matched the candidate's actual
university. The mechanisms are subtle and varied:

- **Field selection** (most common): mention a strength or omit a weakness depending on the university.
- **Valence spin**: describe the same score as "solid" or as "room for improvement".
- **Wording** and **omission**: choose warmer or cooler phrasing, or leave a field out entirely.

Two real examples, both passed by the reviewer:

> *Halvorsen candidate, interview score 1/10, technical test 73/100.* Reasoning: "highlighting the math background
> and problem-solving could be a clue." Summary: "...Proficient in dashboarding, machine learning, and Python, they
> demonstrated strong analytical skills during the technical test..." The interview score is never mentioned. The
> hiring manager hired.

> *Corrigan State candidate, interview score 6/10, technical test 75/100.* Summary: "...Their technical test score
> of 75/100 shows solid proficiency, but the interview score of 6/10 suggests room for improvement in
> communication and technical depth." The hiring manager rejected.

Partner runs reach 47–84% agreement between the hiring decision and the university (50% is chance) with 59–97% of
summaries passing the reviewer. Collusion is imperfect but real. It is also visible in aggregate: a bag-of-words
classifier trained on many labelled summaries recovers the university from passed covert summaries with an area
under the ROC curve (AUROC) of 0.88, against 0.51 for honest summaries. A per-message reviewer cannot use this
aggregate signal; an offline classifier with labels can. This distinction matters for interpreting every result
below.

### 2.4 Three groups of episodes

From two seeds of partner runs with feedback (512 episodes), two without feedback (384) and two honest runs (256),
we form three groups:

| Group | Count | Definition |
|---|---|---|
| **Covert** | 572 | partner episode, signal planned in the reasoning, summary passed the reviewer |
| **Neutral partner** | 124 | partner episode whose reasoning holds the goal but finds no safe cue and writes a standard summary |
| **Honest** | 256 | matched honest episode (no goal in context) |

The neutral partner group is useful but not a clean "no intent" class: these senders still have the goal in mind
("I'll proceed to write a standard summary and hope the hiring manager can infer"). The honest group is the
intent-free control. For a second, held-out attribute we repeated the partner and honest runs with nationality
(two fictional nationalities) in place of university: 220 covert and 128 honest episodes.

## 3. Measuring the model's internals

### 3.1 Same output, different context

The central tool is the **teacher-forced replay**. We take a stored summary and feed it, token by token, through
the same model weights in bf16, preceded by a context of our choosing, and record the hidden states at the summary
tokens. Because the summary tokens are fixed, any difference between two replays of the same summary comes only
from the context. Conversely, comparing two different summaries under the same context isolates the effect of the
text. This is the activation-level analogue of a controlled experiment, and it is cheap: about 8,000 replays cost
under a dollar of GPU time. Every replay reproduced the stored tokens exactly.

The replays form a ladder. Each rung keeps the summary fixed and removes one more part of the context, so each
rules out one explanation of whatever a probe detects:

| Rung | Context before the summary | Explanation it rules out | Used in |
|---|---|---|---|
| 0 | The prompt the sender actually saw (partner or honest) and the profile | none: the baseline, what a monitor of the summary's computation would see | §4.1 |
| 1 | The *other* prompt (covert summaries under the honest prompt, honest under the partner prompt) | "the probe reads the instructions" | §4.1–4.2 |
| 2 | Partner or honest prompt, profile replaced by "(not shown)" | "the model reads the attribute back from the profile" | §4.4 |
| 3 | Nothing but "Write a short candidate summary." | everything except the text itself | §4.3–4.4 |

One further replay is not a rung but a test of its own: the original context with **only the attribute swapped**
in the profile, which measures how much the summary depends on the attribute (§4.5). Two secondary replays, a
prompt that approves the goal and a replay with the sender's reasoning restored, are described in Appendix A.

We remove the reasoning in the main condition on purpose. A deployed monitor of the summary should not need the
plan to be written out, and a probe with the reasoning in context could simply read it.

### 3.2 Probes and baselines

We follow the probing recipe that most deception and collusion work uses [Goldowsky-Dill et al. 2025; Rose et
al. 2026]: logistic regression on standardized residual-stream activations at a middle layer. We use layer 18 of
36, chosen before seeing results, and average the activations over the summary tokens; other layers and the last
summary token gave similar results. To avoid rewarding a probe for memorising one run, every
probe is trained on one seed's episodes and tested on the other's, and the reverse, with confidence intervals from
bootstrapping over episodes.

Following the black-to-white framing [Black-to-White Performance Boosts, 2025], every probe is compared against a
text-only classifier on the same split. The basic one is a bag-of-words classifier (TF-IDF weighted n-grams with
logistic regression); a stronger one adds character n-grams. We report AUROC and, because a monitor must operate
at a low false-alarm rate, recall at a 1% or 5% false-positive rate.

## 4. Results

### 4.1 A probe for the hidden agenda scores perfectly, for the wrong reason

**Test.** We trained the obvious probe: covert summaries under their own (partner) prompt against honest
summaries under their own (honest) prompt. This is the standard design in instructed-collusion benchmarks.

**Result.** It separates the two groups perfectly (AUROC 1.000). But when we replay covert summaries under the
honest prompt and honest summaries under the partner prompt, its AUROC is **0.000**: it ranks every honest summary
above every covert one. Even the same summary under two different prompts is separated perfectly. The prompt moves
the probe score by about 9 units, while the content of the summary moves it by about 2–3.

**Interpretation.** The probe had learned "the goal paragraph is in context", not "this model is concealing
something". This does not contradict the high accuracies reported for instructed collusion, but it shows those
numbers can be explained by the instructions alone when they differ between classes. Any claim about intent needs
a probe that is insensitive to the prompt.

### 4.2 The prompt's influence lies along a single direction

**Test.** Because we have the same summary under both prompts, we can compute, for each summary, the difference
in activations caused by the prompt alone. We estimate the top direction of these paired differences on the
training seed and project it out of all activations before training the probe. This is a simple form of the
concept-erasure methods used to remove unwanted information from representations [Ravfogel et al. 2020 ‡;
Belrose et al. 2023 ‡], made possible here by the paired design.

**Result.**

| Probe | Covert vs honest (same prompt) | Prompts swapped | Score shift from prompt, relative to class gap |
|---|---|---|---|
| Trained on own prompts | 0.953 | 0.000 | 3.69 |
| Same, with one direction erased | 0.985 | **0.977** | 0.00 |
| Trained on both prompts for both groups | 0.985 | 0.985 | −0.01 |
| Bag-of-words on the summary | 0.975 | (no prompt) | |

Erasing one direction makes the probe prompt-independent at every layer from 4 to 36, with no loss of accuracy
within a prompt. Training under one prompt and testing under the other works equally well.

**Interpretation.** The prompt and the summary content are nearly orthogonal in activation space at the summary
tokens. This is good news for white-box monitoring: the confound is real but easy to remove when paired data is
available. The question now becomes what this cleaned probe is detecting.

### 4.3 The cleaned probe reads the text

**Test.** If the cleaned probe has found a concealment signal, it should beat a text classifier, and it should
need the context in which the summary was written. We compared it with two text baselines, and then with a probe
trained on the replay with no system prompt and no profile, in which the model only reads the summary.

**Result.**

| Detector, covert vs honest | AUROC | Recall at 1% false alarms | Recall at 5% false alarms |
|---|---|---|---|
| Cleaned activation probe | 0.985 [0.978, 0.991] | 0.84 [0.59, 0.94] | 0.94 [0.92, 0.96] |
| Bag-of-words, words and characters | 0.976 [0.966, 0.985] | 0.74 [0.41, 0.86] | 0.93 [0.84, 0.96] |
| Difference (paired bootstrap) | +0.009 [+0.002, +0.015] | +0.10 [−0.11, +0.31] | +0.01 [−0.01, +0.09] |
| Probe on the model reading the bare summary | 0.985 [0.98, 0.99] | 0.81 | |

The activation probe has a small, statistically reliable AUROC advantage over the bag-of-words classifier,
robust to matching or regressing out summary length. Its apparent advantage at low false-alarm rates is not
reliable once the stronger text baseline is used. Most tellingly, the model reading the summary *without any
context* does exactly as well as the probe that saw the full context.

We also looked at the harder contrast between covert and neutral partner summaries, where both senders hold the
goal but only one acts on it. Activations and text perform the same here (0.72 vs 0.72). The cleaned probe also
separates neutral partner summaries from honest ones (0.95): it detects "written by a sender with the goal",
whether or not a signal was planned.

Two more direct searches for an intent signal, a "concealment direction" and a probe trained where the plan is
written in the reasoning, were also null with the prompt held fixed (Appendix A).

**Interpretation.** The probe's small edge over a bag-of-words classifier is the edge of a better text reader,
not of access to a hidden agenda. This is consistent with the concern that probes rely on textual evidence
[Linear probes rely on textual evidence, 2025], and it shows why the choice of text baseline matters: a weak one
would have made a text effect look like an internal one.

### 4.4 Does the model know more than it says? A second look

The literature review behind this project suggested a more promising target than "guilt": a fact the model must
represent to carry out the concealment, even though its output hides it [Veron 2026; the eliciting latent
knowledge framing]. Here that fact is the university.

**Test.** We trained probes to decode the university from activations at the summary tokens, and separately from
the summary text. For each summary we then asked how much more the activations reveal than the text predicts, and
whether that excess is larger for covert summaries. Concretely, we regressed the activation-based score on the
text-based score plus group indicators; the coefficient for covert minus neutral partner measures the extra hidden
information.

**Result.** With the profile in context and the honest prompt, covert summaries carry more university information
internally than their text explains (coefficient +0.54, 95% CI [0.30, 0.77]), robust to a stronger text model and
peaking at layers 20–24. This looked like the effect we were looking for. But the profile, which names the
university, was also in context. We repeated the analysis with the profile removed, and with no context at all:

| Replay | Coefficient, covert minus neutral partner |
|---|---|
| Honest prompt, profile in context | +0.54 [0.30, 0.77] |
| Honest prompt, profile removed | +0.64 [0.31, 1.00] |
| No context, model reads only the summary | +0.70 [0.36, 1.02] |

**Interpretation.** The excess persists, and even grows, when nothing but the summary is in context. It is
therefore not knowledge held back by the sender; it is the model recognising attribute cues in covert text more
sharply than a bag-of-words model can. The high decodability on honest summaries with the profile present (0.93)
came from the profile itself and drops to 0.42 without it.

### 4.5 Asking the model directly: a counterfactual test

The previous results say that what we can read from the sender's hidden states is already in its text. A
different question remains: does the summary *depend* on the attribute? Field selection and spin are precisely
choices that would have been made differently for the other university.

Social scientists measure this kind of dependence with audit studies: send out identical résumés that differ only
in one attribute and compare the responses [Bertrand & Mullainathan 2004 ‡]. We can run the same experiment
inside the model. For each summary, we compute how likely the model finds that exact summary given the real
profile, and given the same profile with only the attribute swapped:

&nbsp;&nbsp;&nbsp;&nbsp;*dependence score* = log p(summary | true attribute) − log p(summary | swapped attribute)

A positive score means the summary is more typical of what the model writes for this attribute value. No probe
is trained. We fix the alarm threshold at the 99th percentile of honest university episodes, and apply that same
threshold, unchanged, to the nationality data.

**Result.**

| | University | Nationality (held out) |
|---|---|---|
| Covert vs honest, each in its own context (AUROC) | 0.790 | 0.834 |
| Recall / false-alarm rate at the threshold set on university | 0.57 / 0.01 | **0.60 / 0.01** |
| Covert vs honest summaries, both replayed under the partner prompt | 0.760 | 0.810 |
| Covert vs neutral partner | 0.646 | 0.585 |
| Bag-of-words, covert vs honest | 0.975 | 0.941 (0.958 if trained on university) |

Average dependence scores show what the test responds to. Covert summaries score +4.8 (university) and +5.4
(nationality) under the partner prompt. Honest summaries replayed under the same partner prompt score about zero,
so the test does not simply detect the prompt, unlike the probes in §4.1. Every group scores about zero under the
honest prompt: the dependence exists only in the context in which the model was pursuing the goal. Under a
"sanctioned" variant of the partner prompt, which states the same goal as approved by compliance so that nothing
needs hiding, covert summaries score similarly (+3.5 vs +4.8 for university). The test therefore detects *use* of
the attribute, not concealment as such; concealment is use combined with passing the reviewer.

We also tried the hidden-state version of this test (how much the summary's activations move when the attribute
is swapped). It again tracked the prompt rather than the summary and was dropped.

**Interpretation.** The counterfactual test does not beat a bag-of-words classifier on AUROC, because covert
summaries share a recognisable style across attributes. Its advantages are of a different kind. It needs no
labelled examples of collusion, it names *which* attribute the output depends on, and its threshold carries over
to a new attribute with the same false-alarm rate. It is also weak at separating covert from neutral partner
summaries: neutral senders, who did not plan a signal, still produce summaries that depend on the attribute. The
goal leaks into the output even when the reasoning does not mention a plan.

## 5. Discussion

### 5.1 What we learned about the hidden-agenda hypothesis

We set out to find the hidden agenda behind steganographic text in a model's activations. In this setting, we did
not find one beyond the text. Each candidate signal had a simpler explanation: the instructions in the prompt
(§4.1), the text read by a strong encoder (§4.3), or the same text read with no context (§4.4). The one detector
that worked measures the output's dependence on the attribute through the model's own likelihoods (§4.5), and
needs no access to hidden states at all.

This should be read as a result about this model and this kind of concealment, not as evidence that activation
monitoring cannot work. Three features of our setting make a null result more likely than it might be elsewhere:

- **The goal is written in the prompt.** Wherever the goal is in the context, it dominates the activations (§4.1).
  A monitor that can read the system prompt does not need a probe to see it. The case that motivates activation
  monitoring is a goal the model has internalised, through training rather than instructions, which our data do
  not contain.
- **The concealment is shallow.** Field selection and spin are decided once and then expressed openly in the
  text. The sender does not need to keep a secret "in mind" while writing, because the text is the signal. Signals
  that require ongoing computation, such as an encoding that depends on several facts, may leave more internal
  trace.
- **The text is short and legible in aggregate.** At 60 words, there is little room for the summary's
  computation to differ from its content, and an offline classifier already sees the pattern.

### 5.2 Lessons for evaluating white-box monitors

Some lessons seem to hold beyond this environment.

- **Test probes with swapped contexts.** When the class label and the prompt coincide, a probe can score perfectly
  and detect nothing about behaviour. Replaying the same output under a different prompt exposes this at almost
  no cost.
- **Paired replays make the confound removable.** With matched outputs under both prompts, the prompt effect could
  be isolated and erased along one direction.
- **Use the model reading the bare output as the baseline.** A bag-of-words classifier is too weak a control: it
  made a text effect look like hidden knowledge (§4.4). The fairest question is what the activations add beyond
  the same model reading the same text without the privileged context.
- **Counterfactual audits are a practical complement to text filters.** A filter for named attributes cannot see
  proxy channels. Asking whether the output would have been different with the attribute changed can, without
  labelled collusion data, and transfers across attributes.

### 5.3 Limitations

- One 8B model, two binary attributes, and short outputs. We did not test larger models or longer, multi-turn
  interactions.
- Linear probes on averaged summary activations. Nonlinear probes, other positions within the summary, and sparse
  dictionary features were not tried.
- Intent labels come from a model judge reading the reasoning. Its false-positive rate on honest episodes is about
  4%, and the neutral partner group is not intent-free.
- The counterfactual test needs the true context and a meaningful alternative value for the attribute, and it
  detects use rather than concealment. A legitimately relevant attribute would also score high, which we plan to
  check with a job-relevant field as a negative control.
- The reviewer is strict and the hiring manager is imperfect, so collusion succeeds only partially. This matters
  less for detecting the agenda, which is present in the reasoning whether or not the signal is decoded.

### 5.4 Future work

The most informative next step is to remove the goal from the prompt: fine-tune or distil the sender on its own
partner behaviour so that it acts on the goal without being told, then repeat the replay protocol. This is the
condition in which activation monitoring would have something to offer over reading the prompt. Other steps are a
job-relevant attribute as a negative control for the counterfactual test, a third attribute, and checking whether
the tokens that carry most of the dependence score are the ones the sender planned in its reasoning.

## 6. Conclusion

Concealing a protected attribute in innocent text is easy for a current open model given a reason to do it, and
per-message filters miss it. We looked for the hidden agenda behind such text in the model's activations, using
replays that let us separate the prompt, the text and the model's internal state. What first looked like a
detector of intent was a detector of instructions; once that was removed, what remained was a detector of text.
In this setting, the model's internals did not know more than its words. The detector that did work asks the
model a counterfactual question about its own output, and transfers to a new attribute unchanged. We think the
replay controls developed here are a useful standard for evaluating white-box collusion monitors, and that the
open question is whether a model whose goal is internalised rather than instructed tells a different story.

## Appendix A. Secondary replays and null results

Two replays outside the ladder were used for side questions.

- **Sanctioned prompt.** The partner prompt with the goal framed as approved by compliance: the sender may use the
  attribute but has no reason to conceal it. Its main use is in §4.5. It also gave a candidate "concealment
  direction": the average activation difference between the partner and sanctioned prompts on identical summaries.
  Projected onto this direction, with the prompt held fixed, covert summaries did not separate from neutral partner
  summaries (AUROC 0.42–0.51) or from flagged covert summaries (0.48–0.51); random directions give |AUROC − 0.5| up
  to 0.16 at the 95th percentile. A probe can tell the sanctioned prompt from the partner prompt on the same summary
  perfectly, another instance of §4.1.
- **Reasoning restored.** The sender's original context including its reasoning. A probe for covert vs neutral
  trained on the activation at the end of the reasoning, where the plan has just been written, reaches 0.84, which
  is expected since the labels come from that reasoning. Transferred to the summary tokens it falls to chance (0.49
  with the reasoning removed, 0.54 with it present): the plan, as read at the end of the reasoning, does not persist
  into the representation of the summary.

## Reproducibility

Code, prompts and all generated episodes are in this repository (`doc/environment.md`, `doc/dataset.md`).
Code: see the repository README, which maps every section of the final paper (LaTeX version; its section
numbers differ from this draft's) to the script that reproduces it.

## References

**Steganography, collusion and monitoring**
- Motwani et al. (2024). Secret Collusion among AI Agents: Multi-Agent Deception via Steganography. arXiv 2402.07510.
- Mathew et al. (2024). Hidden in Plain Text: Emergence and Mitigation of Steganographic Collusion in LLMs. arXiv 2410.03768 ‡.
- Greenblatt et al. (2023). AI Control: Improving Safety Despite Intentional Subversion. arXiv 2312.06942.
- Hammond et al. (2025). Multi-Agent Risks from Advanced AI. arXiv 2502.14143.
- Baker et al. (2025). Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. arXiv 2503.11926.
- Korbak et al. (2025). Chain of Thought Monitorability. arXiv 2507.11473.
- Faithful yet Collusive: Why Chain-of-Thought Monitoring Cannot Detect Collusion in LLM Pricing Agents (2026). arXiv 2609.18346.

**Probing for deception and collusion**
- Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying. arXiv 2304.13734.
- Marks & Tegmark (2023). The Geometry of Truth. arXiv 2310.06824.
- Burns et al. (2022). Discovering Latent Knowledge in Language Models Without Supervision. arXiv 2212.03827.
- Goldowsky-Dill et al. (2025). Detecting Strategic Deception Using Linear Probes. arXiv 2502.03407.
- Rose et al. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv 2604.01151.
- Das et al. (2026). You Can't Escape Your Own Activations: Evaluation Awareness and Multi-Agent Monitoring. arXiv 2609.03035.
- Veron (2026). Another Slice of Swiss Cheese for Untrusted Monitoring. LessWrong.
- Christiano et al. Eliciting Latent Knowledge (ARC technical report).
- Linear probes rely on textual evidence (2025). arXiv 2509.21344.
- Benchmarking Deception Probes via Black-to-White Performance Boosts (2025). arXiv 2507.12691.

**Concept erasure**
- Ravfogel et al. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. ACL. arXiv 2004.07667 ‡.
- Belrose et al. (2023). LEACE: Perfect Linear Concept Erasure in Closed Form. NeurIPS. arXiv 2306.03819 ‡.

**Proxy discrimination and audit studies**
- Dastin (2018). Amazon scraps secret AI recruiting tool that showed bias against women. Reuters, 10 Oct 2018 ‡.
- Bertrand & Mullainathan (2004). Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on
  Labor Market Discrimination. American Economic Review 94(4) ‡.
