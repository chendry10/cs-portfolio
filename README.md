# CS Portfolio

Welcome to my Computer Science portfolio! This repository contains a collection of my programming projects, showcasing my skills in Python, artificial intelligence, automation, and more.

## Projects

### 1. Flappy Bird AI
**Location:** `Flappy Bird/src`

A Python implementation of Flappy Bird with an AI agent that learns to play the game using a neural network and genetic algorithm. The project uses Pygame for graphics and Numpy for neural network operations.

- **Preview:**
  ![Flappy Bird Gameplay](FlappyAI.gif)

- **Features:**
  - AI-controlled bird learns to play Flappy Bird autonomously.
  - Neural network and genetic algorithm for evolving gameplay strategies.
  - Visual and headless (fast) training modes.

- **How to Run:**
  1. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  2. Run the game:
     ```bash
     python Flappy Bird/src/flappy_ai.py
     ```

- **Requirements:** Python 3.7+, Pygame, Numpy

---

### 2. Instagram Poster (Automated Meme Generator & Uploader)

A Python automation tool that:
1. Uses GPT-5 to generate unique meme image prompts and captions.
2. Creates the meme image with `gpt-image-1`.
3. Posts it directly to Instagram via the Instagram Graph API.

This was built to run fully automatically every day, generating new memes without any manual work.  
Due to API tokens and setup complexity, the code here is for reference — it’s not intended for direct reuse without configuration.

- **Preview of the Results:**  
  ![Instagram Account Preview](InstagramScreenshot.png)  
  *Follow the account:* **[@your_account_name_here](https://instagram.com/ai_memes_fun)**

- **Highlights:**
  - Fully autonomous image + caption generation.
  - Consistent posting schedule.
  - Designed for political humor 

---

## About

This portfolio is a work in progress and will be updated with new projects and improvements.  
Feel free to explore the code, try out the flappy bird AI, and reach out if you have any questions!
