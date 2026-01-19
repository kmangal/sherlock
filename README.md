# Sherlock

Sherlock identifies potential cheating on multiple choice exams using statistical analysis.

## How It Works

Sherlock looks for unusually high numbers of matching answers between test-takers and flags suspicious cases for investigation. While matching answers don't prove cheating occurred, they provide a focused starting point for further review.

Determining what counts as "unusually high" depends on the exam and test-takers. For instance, easy questions that everyone answers correctly create legitimate matches. The key question is: What's the likelihood that two candidates of similar ability would naturally answer the same questions identically?

To answer this, Sherlock builds a statistical model from the answer patterns, then simulates how often candidates would have similar answers based purely on their ability and question difficulty. Using these simulations, it calculates the probability of observing the actual degree of similarity and flags cases where this probability falls below a set threshold.

## Getting Started

See CONTRIBUTING.md for a guide to getting your local development environment set up.