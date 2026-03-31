# Salary Predictor using Linear Regression

A simple machine learning project that predicts an employee's salary based on their years of experience. Built as a first-year AI & ML college project to understand how linear regression works in practice.

---

## What does this project do?

It takes years of work experience as input and predicts an approximate salary using a Linear Regression model. The model is trained on a small sample dataset, and once trained, it can predict the salary for any number of years of experience you give it.

---

## How does it work?

Linear Regression draws the best-fit straight line through the data points. Once we have that line, we can use it to predict salary values for new inputs. The formula is simple:

```
Salary = (slope × experience) + intercept
```

The model learns the values of slope and intercept automatically from the training data.

---

## Requirements

Make sure you have Python installed (version 3.7 or above). Then install the required libraries by running:

```bash
pip install numpy scikit-learn matplotlib
```

That's it — no other setup is needed.

---

## How to run it

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Run the script:

```bash
python salary_predictor.py
```

4. When prompted, enter the number of years of experience:

```
Enter years of experience to predict salary: 5
Predicted salary for 5.0 years of experience: ₹55,000.00
```

5. A plot will also be generated and saved as `salary_vs_experience.png` in the same folder.

---

## Project Structure

```
salary-predictor/
│
├── requirements.txt         # Requirements
├── salary_predictor.py        # Main Python script
├── salary_vs_experience.png   # Plot generated after running the script
├── project_report.pdf         # Project Report 
└── README.md                  # This file
```

---

## Sample Output

After running the script, you will see something like this in the terminal:

```
=== Sample Dataset ===
Experience (yrs)     Salary (INR)
-----------------------------------
1                    30000
2                    35000
...

=== Model Training Complete ===
Slope (coefficient) : 6933.33
Intercept           : 23066.67

=== Model Evaluation ===
Mean Absolute Error : 1500.00 INR
R² Score            : 0.9985

=== Salary Prediction ===
Enter years of experience to predict salary: 6
Predicted salary for 6.0 years of experience: ₹64,666.65
```

---

## Notes

- The dataset used here is a small hardcoded sample just for learning purposes.
- In a real project, you would replace this with actual data loaded from a CSV file.
- The model works best when the relationship between experience and salary is roughly linear.

---

## Author
**Abhishek Singh**  
**B.Tech. CSE (AI & ML)**  
Made with the help of Python, scikit-learn, and a lot of trial and error.
