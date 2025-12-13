# Movie Recommendation System using Matrix Factorization

## Overview
This project implements a movie recommendation system using the MovieLens dataset.
A baseline model is compared against a Matrix Factorization model (SVD) to demonstrate performance improvement.

## Dataset
- MovieLens Latest Small
- ~100k ratings
- User–movie explicit feedback

## Approach
1. Exploratory data analysis
2. Global mean baseline
3. Matrix Factorization using SVD
4. Evaluation with RMSE
5. Top-N movie recommendations

## Results
| Model        | RMSE  |
|-------------|-------|
| Baseline    | 1.049 |
| SVD         | 0.881 |

## Example Recommendation
Top recommendations are generated for a given user based on predicted ratings.

## Tech Stack
- Python
- pandas, numpy
- scikit-surprise
- scikit-learn

## Limitations
- Cold-start problem for new users/items
- Predicted ratings may saturate at upper bound
- Uses explicit feedback only

## Future Work
- Add content-based filtering
- Handle cold start users
- Build FastAPI endpoint
