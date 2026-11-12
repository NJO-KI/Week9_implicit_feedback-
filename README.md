# DSA 4060: Recommender Systems — Week 9 Guided Lab
## From Implicit Events to Ranked Recommendations with Confidence-Weighted ALS

### Lab Overview
This repository contains the code, analysis, and deliverables for the **Week 9 Guided Lab** in **DSA 4060 (Recommender Systems)**. The lab explores the transition from explicit ratings to **implicit feedback** (views, plays, saves) and implements a confidence-weighted Alternating Least Squares (ALS) matrix factorization model using Python, SciPy, Pandas, and the `implicit` library in Visual Studio Code.

---

### Key Pipeline Stages
1. **Event Log Generation:** Simulates a music-streaming interaction dataset with actions (`play`, `repeat_play`, `save`).
2. **Interaction Strength:** Maps event types to custom numerical weights to quantify engagement.
3. **Preference & Confidence Transformation:** Separates binary preference ($p_{ui}$) from confidence ($c_{ui} = 1 + \alpha r_{ui}$).
4. **Sparse Matrix Construction:** Builds an efficient Compressed Sparse Row (CSR) matrix.
5. **ALS Model Training:** Trains user and item latent factor embeddings.
6. **Top-N Ranking:** Generates personalized Top-5 recommendations for target users.
7. **One-Variable Experiments & Evaluation:** Explores hyperparameter sensitivity ($\alpha$, factors, regularization) and time-aware evaluation design.

---

### Lab Exit Ticket Reflection (~180 Words)

When converting raw interaction events into numerical weights, we make the key assumption that event types serve as fixed linear proxies for true user preference—implying, for instance, that a single save reliably equals five plays regardless of active context or autoplay bias. 

The core distinction between preference and confidence is that **preference** ($p_{ui}$) is binary (answering whether any interaction was observed, e.g., $p_{ui}=1$ for a single play), while **confidence** ($c_{ui} = 1 + \alpha \cdot r_{ui}$) scales with observed strength (a user with 1 play has confidence $c_{ui} = 21$, whereas a user with 5 saves has $c_{ui} = 501$, instructing ALS to weight that observation higher).

In our parameter experiment, increasing $\alpha$ from 5 to 40 shifted the Top-5 recommendations toward higher-frequency catalog tracks by aggressively penalizing unobserved candidate items. 

A primary limitation of this matrix-factorization baseline is the **cold-start problem** for brand new items lacking interaction history. A key next improvement is incorporating side metadata via a hybrid architecture like LightFM.

---
