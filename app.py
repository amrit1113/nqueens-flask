from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import os

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend communication

class PSO:
    def __init__(self, n, swarm_size=30, max_iter=1000, w=0.5, c1=1.5, c2=1.5):
        self.n = n
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.swarm = [self.init_particle() for _ in range(self.swarm_size)]
        self.velocities = [self.init_velocity() for _ in range(self.swarm_size)]
        self.best_positions = self.swarm[:]
        self.best_scores = [self.fitness(p) for p in self.swarm]
        self.global_best_position = min(self.best_positions, key=lambda p: self.fitness(p))
        self.global_best_score = min(self.best_scores)

    def init_particle(self):
        return random.sample(range(self.n), self.n)

    def init_velocity(self):
        return [random.uniform(-1, 1) for _ in range(self.n)]

    def fitness(self, position):
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if position[i] == position[j] or abs(position[i] - position[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    def update_position(self, i):
        for j in range(self.n):
            self.swarm[i][j] = int(self.swarm[i][j] + self.velocities[i][j]) % self.n
        self.swarm[i] = self.ensure_no_conflicts(self.swarm[i])

    def update_velocity(self, i):
        for j in range(self.n):
            r1, r2 = random.random(), random.random()
            self.velocities[i][j] = (self.w * self.velocities[i][j] + 
                                     self.c1 * r1 * (self.best_positions[i][j] - self.swarm[i][j]) +
                                     self.c2 * r2 * (self.global_best_position[j] - self.swarm[i][j]))

    def ensure_no_conflicts(self, position):
        seen = set()
        for i in range(self.n):
            while position[i] in seen:
                position[i] = random.randint(0, self.n - 1)
            seen.add(position[i])
        return position

    def optimize(self):
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            for i in range(self.swarm_size):
                current_fitness = self.fitness(self.swarm[i])
                if current_fitness < self.best_scores[i]:
                    self.best_scores[i] = current_fitness
                    self.best_positions[i] = self.swarm[i]
                if current_fitness < self.global_best_score:
                    self.global_best_score = current_fitness
                    self.global_best_position = self.swarm[i]
            if self.global_best_score == 0:
                break
            for i in range(self.swarm_size):
                self.update_velocity(i)
                self.update_position(i)
        return self.global_best_position, self.global_best_score, iteration

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    n = int(request.json['n'])
    pso = PSO(n)
    solution, conflicts, iterations = pso.optimize()
    return jsonify({'solution': solution, 'conflicts': conflicts, 'iterations': iterations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
