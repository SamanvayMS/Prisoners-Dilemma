# Axelrod's Tournament in OpenAI Gym

## Overview
This project implements a custom environment for Axelrod's Tournament within the OpenAI Gym interface. It focuses on the application of advanced machine learning techniques, specifically a Boltzmann Machine and a Deep Neural Network, to develop and learn competitive strategies in an iterated prisoner's dilemma setting.

## My Agents
- **Boltzmann Machine Agent**: Utilizes a Boltzmann machine for strategy optimization based on reward maximization and temperature-based probability distribution.
- **Deep Neural Network Agent**: Employs a deep learning architecture to learn and adapt strategies through experience and pattern recognition.

## Agents from the Paper
### 1. Tit-for-Tat (TFT)
- **Description**: Begins with cooperation and then simply mimics the opponent's previous move.
- **Algorithm**:
  ```python
  if first_round:
      return Cooperate
  else:
      return opponent_previous_move
  ```
### 2. Generous Tit-for-Tat (GTFT)

- **Description**: Similar to TFT, but occasionally forgives a defection.
- **Algorithm**:
  ```python
  if first_round:
      return Cooperate
  elif opponent_previous_move == Defect:
      return Cooperate with small probability
  else:
      return Cooperate
  ```
### 3. Tit-for-Two-Tats (TF2T)

- **Description**: Cooperates unless the opponent defects twice in a row..
- **Algorithm**:
  ```python
  if first_round or second_round:
      return Cooperate
  elif opponent_last_two_moves == [Defect, Defect]:
      return Defect
  else:
      return Cooperate
  ```

### 4. Win-Stay-Lose-Shift (WSLS)

- **Description**: Continues the previous action if it was successful; otherwise, switches.
- **Algorithm**:
  ```python
  if first_round:
      return Cooperate
  elif last_round_successful:
      return previous_move
  else:
      return opposite_of_previous_move
  ```

### 5. Always Cooperate (ALLC) and Always Defect (ALLD)

- **ALLC Description**: Cooperates in every round.
- **ALLD Description**: Defects in every round.
- **Algorithm**:
  ```python
  # For ALLC
  return Cooperate
  
  # For ALLD
  return Defect
  ```

### 6. Random

- **Description**: Chooses randomly between cooperation and defection.
- **Algorithm**:
  ```python
  return random_choice([Cooperate, Defect])
  ```

### 7. Grim

- **Description**: Starts cooperating and then defects forever after the first defection by the opponent.
- **Algorithm**:
  ```python
  if opponent_defected_before:
      return Defect
  else:
      return Cooperate
  ```

### 8. Hard Tit-for-Tat (Hard TFT)

- **Description**: A stricter version of TFT, less likely to return to cooperation after defection.
- **Algorithm**:
  ```python
  if first_round:
      return Cooperate
  elif opponent_previous_move == Defect:
      return Defect with higher probability than TFT
  else:
      return Cooperate
  ```

### 9. Extort-2 and ZDGTFT-2 (Zero-Determinant Strategies)

- **Description**: Strategies that enforce a specific relationship between the player’s and the opponent’s scores.
- **Extort-2 Algorithm**:
  ```python
  # Based on complex mathematical formula to ensure player outscores opponent
  ```
- **ZDGTFT-2 Algorithm**:
  ```python
  # A variant that allows for some cooperation while maintaining scoring advantage
  ```


## Environment
The environment is modeled after the classical Axelrod's Tournament with a focus on iterated prisoner's dilemma games. The environment supports multiple agent interactions and records strategies, moves, and payoffs for in-depth analysis.

## Installation

```bash
git clone [your-repository-url]
cd [your-repository-name]
pip install -r requirements.txt
```
## Usage

To run the tournament:

```bash
python run_tournament.py
```
## Strategy Learning

* Dynamic Programming Strategy: Utilizes dynamic programming for optimizing the decision-making process based on the history of plays and anticipated future moves.
Learning Mechanism: Both the Boltzmann machine and the DNN agent continuously learn and adapt their strategies based on game outcomes and opponent behaviors.
Experimentation

* Experiment with different strategies and learning rates. The framework allows for easy integration of new agents and strategies for comprehensive analysis and experimentation.

## Results and Analysis

Results of the tournaments, including strategy effectiveness, win/loss ratios, and learning progression, are recorded for detailed analysis. Visualization scripts are provided for graphical representation of the tournament outcomes.

## Contributing

Contributions to the project are welcome. Please follow the standard pull request process for contributions.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

