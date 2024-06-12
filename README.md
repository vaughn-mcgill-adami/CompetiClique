# CompetiClique
A simple game on graphs solved using reinforcement learning.

Rules:

Let k, m, and n be positive integers.

Two players, the builder and the forbidder, take turns modifying a graph. 

The builder's goal is to build a k vertex clique in a graph.
The forbidder's goal is to prevent this from occurring as long as possible.

We start from an empty graph. First, the builder places m edges such that none of the edges have both of their endpoints in the forbidden set. Then, the forbidder selects n vertices to include in the forbidden set. Repeat until the builder wins. The forbidden set and the graph grow throughout the game. 