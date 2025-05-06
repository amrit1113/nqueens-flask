# 🧠 N-Queens Solver using Particle Swarm Optimization (PSO)

This is a web-based application that solves the N-Queens problem using the **Particle Swarm Optimization (PSO)** algorithm. Built using **Flask**, it allows users to input any number of queens (`n`) and visualizes the solution on an `n × n` chessboard.

## 🚀 Features

- Solve the classic N-Queens problem using PSO metaheuristic.
- Dynamic input: users can choose the number of queens.
- Interactive and user-friendly interface.
- Real-time solution visualization on the chessboard.
- Deployed on **Render** for easy online access.

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Algorithm:** Particle Swarm Optimization (PSO)  
- **Deployment:** Render


## 🧩 How PSO Works for N-Queens

- Each particle represents a possible configuration of queens.
- The fitness function evaluates how many queens are in conflict.
- Particles update positions based on personal and global bests.
- Iteratively improves the solution until an optimal or near-optimal configuration is found.

