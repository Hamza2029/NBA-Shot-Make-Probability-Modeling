# NBA Shot Make Probability Modeling

This project develops a machine-learning model that estimates the probability that an NBA field-goal attempt is made. Using more than 425,000 labeled shots, I analyzed how shot geometry, defensive pressure, player movement, possession context, and player identity contribute to shot difficulty.

The final CatBoost model reduced log-loss by **9.8%** relative to a constant-probability baseline and produced well-calibrated probabilities on an untouched validation set.

## Project Overview

Rather than predicting only a make or miss, the model outputs a probability for every shot attempt. Calibrated probabilities are useful for applications such as:

- Measuring shot quality independently of the observed result
- Comparing actual and expected shooting performance
- Evaluating offensive decision-making and defensive pressure
- Aggregating expected points across players, lineups, or game situations

The primary evaluation metric was **log-loss**, which rewards accurate probabilities and penalizes confident incorrect predictions.

## Data

The labeled dataset contained **425,719 shot attempts** with information describing:

- Shot type, distance, and court location
- Closest-defender distance and contest information
- Defender positions during the second before release
- Shot clock, dribbles, shooter speed, and game state
- Shooter, defender, team, season, and time context

Before modeling, I audited column types, missingness, duplicates, target balance, categorical cardinality, numeric ranges, and spatial consistency. I also parsed the defender-approach field from string representations of dictionaries into validated time-indexed measurements.

> The original data is not included in this repository.

## Exploratory Analysis

Several patterns shaped the feature design:

- **Shot distance** was the strongest baseline predictor; make rate declined substantially away from the rim.
- **Shot type** remained important after accounting for distance because dunks, layups, floaters, jumpers, and post attempts have different difficulty profiles.
- Greater **closest-defender separation** generally corresponded to higher make rates, although the relationship varied by shot type.
- **Defender motion** provided information beyond static distance, but its univariate relationship was confounded by shot type, initial separation, and defensive rotations.
- **Shooter speed** and other context variables showed composition effects, reinforcing the need for a model that could learn nonlinear interactions.
- Higher-order contester fields were extremely sparse, so I emphasized counts, availability indicators, and distance summaries instead of sparse identities.

## Feature Engineering

I organized 37 features into cumulative basketball-motivated groups:

| Feature group | Examples |
|---|---|
| Shot geometry | Shot type, distance, court coordinates, absolute lateral position, rim angle |
| Defensive pressure | Closest-defender distance, contested flag, number and distances of contesters |
| Defender motion | Time-indexed approach distances, distance covered, average closing speed, recent closing amount |
| Play context | Shot clock, dribbles, shooter speed, game state, three-point indicator |
| Shooter identity | Shooter identifier |
| Defender identity | Closest-defender identifier and related context |
| Team and time | Offensive team, defensive team, season, and temporal context |

This grouping supported cumulative ablation testing, allowing each block to be evaluated by the additional predictive value it contributed.

## Modeling

I selected **CatBoost** because it can:

- Model nonlinear relationships and feature interactions
- Handle numerical and categorical variables in one framework
- Use high-cardinality identity features without manual one-hot encoding
- Apply regularization and early stopping to control overfitting

The evaluation workflow consisted of:

1. Creating an untouched holdout set before model-directed analysis
2. Comparing cumulative feature groups through ablation testing
3. Evaluating finalist feature sets with three-fold cross-validation
4. Using early stopping to estimate the appropriate number of trees
5. Running a focused hyperparameter comparison
6. Evaluating the selected model once on the untouched holdout set

## Results

### Feature ablation

| Cumulative feature set | Features | Validation log-loss | Improvement from prior set |
|---|---:|---:|---:|
| Shot geometry | 7 | 0.642254 | — |
| Defensive pressure | 15 | 0.629857 | 0.012397 |
| Defender motion | 22 | 0.628175 | 0.001683 |
| Play context | 30 | 0.626137 | 0.002038 |
| Shooter identity | 31 | 0.624126 | 0.002011 |
| Defender identity | 33 | 0.622683 | 0.001443 |
| Team and time | 37 | 0.622349 | 0.000333 |

Defensive-pressure features delivered the largest incremental improvement beyond shot geometry. Later feature groups produced smaller but consistent gains.

### Cross-validation and tuning

The complete feature set achieved an out-of-fold log-loss of **0.621739** across three folds, with very little variation between folds. A focused hyperparameter comparison selected:

| Hyperparameter | Selected value |
|---|---:|
| Tree depth | 7 |
| Learning rate | 0.03 |
| L2 leaf regularization | 8 |
| Trees | 1,870 |

### Final holdout performance

| Metric | Result |
|---|---:|
| Constant-probability baseline log-loss | 0.689626 |
| Tuned cross-validation log-loss | 0.621550 |
| Holdout log-loss | 0.621746 |
| Observed holdout make rate | 45.81% |
| Mean predicted probability | 45.60% |

The final holdout score closely matched cross-validation, while the average predicted probability was within 0.21 percentage points of the observed make rate. A probability-bin calibration analysis also closely followed the ideal calibration line.

## Key Takeaways

- Shot geometry establishes the baseline difficulty of an attempt.
- Defensive pressure provides substantial predictive information beyond location and shot type.
- Static separation and recent defender movement capture complementary aspects of pressure.
- Player identity and contextual variables add modest but repeatable predictive value.
- Calibration is essential because the intended output is a probability rather than a class label.

Feature importance describes how the model uses variables for prediction; it should not be interpreted as evidence that changing a feature would causally change shot success.

## Repository Structure

```text
.
├── notebooks/          # Data validation, EDA, feature engineering, and modeling
├── src/                # Reusable preprocessing and modeling code, if separated
├── reports/            # Project writeup and selected figures
├── outputs/            # Predictions and submission files
├── ai_prompts.txt      # Consolidated record of AI-assisted questions
├── requirements.txt    # Python dependencies
└── README.md
```

Adjust this structure to match the final repository before publishing.

## Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add the data locally using the paths expected by the notebooks, then run the notebooks in order.

Main Python libraries include pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and CatBoost.

## Limitations and Future Work

- Use time-based or game-grouped validation to test generalization across games and roster changes.
- Add leakage-safe historical player and team features derived only from prior observations.
- Evaluate calibration separately by shot type, distance, season, and player volume.
- Add grouped permutation importance or SHAP analysis for more detailed interpretation.
- Incorporate richer tracking features such as defender angle, contest hand, shooter orientation, pass origin, and team spacing.

## Author

**Hamza Khalilullah**  
UC Berkeley Data Science
