

-----

# 🤖 Chinook RAG: AI Chat Agent

This project is a powerful AI chatbot built with Streamlit and LangChain. It uses a **RAG (Retrieval-Augmented Generation)** agent to answer questions by intelligently querying multiple data sources:

1.  A **PostgreSQL database** (the Chinook music store).
2.  **Product documentation** (a PDF file).
3.  **Customer feedback** (a TXT file).

The user interface is a simple chat window, but behind the scenes, a powerful AI agent decides the best way to answer your question.

## ✨ Key Features

  * **Conversational AI:** Chat with your data in plain English.
  * **Multi-Source RAG:** The AI can get information from different types of data (SQL, PDF, TXT) to build a complete answer.
  * **LangChain Agent:** Uses a `LangGraph` agent to route questions to the correct tool (SQL query, PDF search, or feedback search).
  * **Dockerized:** The entire application (app and database) is containerized with Docker for easy setup and deployment.
  * **Powered by Gemini:** Uses Google's Gemini models for both understanding language and creating vector embeddings.

## 🧠 Core Concepts Explained

### What is RAG?

**RAG (Retrieval-Augmented Generation)** is a modern AI technique. In simple terms:

  * Instead of the AI *only* using its pre-trained knowledge (which can be old), it first **retrieves** relevant information from your specific documents (like the PDF or database).
  * Then, it uses that freshly retrieved information to **generate** a high-quality, accurate answer.
  * It's the difference between asking a person a question from memory vs. letting them look up the answer in a textbook first.

### What is an "Agent"?

Think of the AI "agent" in this project as a smart manager or a router.

1.  You ask a question, like, "What did customers say about the rock tracks?"
2.  The agent looks at the question and thinks: "This isn't in the product PDF, and it's not a direct database query... it sounds like it's in the customer feedback."
3.  It then "picks up" the correct tool (the feedback search tool) and uses it to find the answer.
4.  If you ask "Who is the composer for the track 'For Those About to Rock'?", the agent will know to use the SQL tool.

## 💻 Technology Stack

  * **Application:** Streamlit, Python 3.10
  * **AI / LLM:** Google Gemini (via LangChain)
  * **Database:** PostgreSQL
  * **Vector Storage:** FAISS
  * **Containerization:** Docker & Docker Compose

-----

## 🚀 Getting Started

You can run this entire project on your local machine or on a cloud server (like an Azure VM) with just one command.

### Prerequisites

1.  **Git:** To clone this repository.
2.  **Docker & Docker Compose:** This is *required*. It will handle installing Python, PostgreSQL, and all dependencies. [Install Docker Desktop](https://www.docker.com/products/docker-desktop/).
3.  **Google API Key:** You need a `GOOGLE_API_KEY` with access to the Gemini models. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Step 1: Clone the Repository

Open your terminal and clone the project:

```bash
git clone https://github.com/nikhilnagar29/ai-assigment.git
cd ai-assigment
```

### Step 2: Create Your Configuration File (Most Important\!)

The application is configured to read its settings from a file named `.env` located inside the `app` folder.

1.  Navigate into the `app` directory:
    ```bash
    cd app
    ```
2.  Create the `.env` file. (You can use `nano .env` on Linux/Mac or `notepad .env` on Windows).
3.  Copy and paste the text below into your new `.env` file.

<!-- end list -->

```ini
# --- Paste your Google API Key ---
GOOGLE_API_KEY=AIzaSy...your...key...here...

# --- Database Connection ---
# This MUST be 'db' to match the service name in docker-compose.yml
DB_HOST=db
DB_PORT=5432
DB_NAME=chinook
DB_USER=chinook
DB_PASSWORD=chinook
```

4.  Save and close the file.
5.  Go back to the main project directory:
    ```bash
    cd ..
    ```

### Step 3: Run the Application\!

Now for the easy part. From the root `ai-assigment` directory (where your `docker-compose.yml` file is), run:

```bash
docker-compose up --build
```

  * `--build`: This tells Docker to build the app image from your `Dockerfile` the first time.
  * The first time you run this, it will take a few minutes. It has to:
    1.  Download the PostgreSQL image.
    2.  Load the entire Chinook database.
    3.  Download the Python image.
    4.  Install all the Python requirements.
    5.  Build the vector stores from your PDF and TXT files.

### Step 4: Open the App

Once you see messages in your terminal that the app is running (e.g., "Streamlit is running on..."), open your web browser and go to:

**[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)**

You should see the chatbot interface\!

-----

## ☁️ Deploying to an Azure Virtual Machine

The steps are almost identical to running it locally.

1.  **Create your Azure VM:**

      * Use an **Ubuntu Server** image.
      * Select your VM size (e.g., `Standard_B2s`).
      * **Networking:** This is critical. In the "Inbound port rules", you MUST allow traffic on:
          * **Port 22** (for SSH, so you can connect)
          * **Port 8501** (for the Streamlit app)

2.  **Connect to your VM:**

    ```bash
    # Use the .pem key you downloaded from Azure
    ssh -i /path/to/your-key.pem azureuser@<YOUR_VM_PUBLIC_IP>
    ```

3.  **Install Docker and Git on the VM:**

    ```bash
    sudo apt update
    sudo apt install docker.io docker-compose git -y
    sudo usermod -aG docker $USER
    # --- IMPORTANT: Log out and log back in for the change to apply ---
    exit
    ```

4.  **Log back in** and follow the *exact* same **Step 1 and Step 2** from the local setup (clone the repo, create the `app/.env` file). The `.env` file contents are identical.

5.  **Run the app in the background:**
    Instead of `docker-compose up`, use the `-d` (detached) flag to run it in the background:

    ```bash
    # Make sure you are in the ai-assigment directory
    docker-compose up --build -d
    ```

6.  **Access your App:**
    Open your browser and go to:

    **`http://<YOUR_VM_PUBLIC_IP>:8501`**

    Your app is now live on the internet\!

## 📂 Project Structure

```
/ai-assigment
├── app/                  # Main application folder
│   ├── core/             # Core logic
│   │   ├── tools/        # RAG tools (feedback, product, sql)
│   │   ├── config.py     # Handles API keys and DB connection
│   │   ├── graph.py      # The LangChain agent logic
│   │   └── vector_builder.py # Script to create vector stores
│   ├── data/             # Raw data for RAG
│   │   ├── feedback/
│   │   └── products/
│   ├── vector_stores/    # Generated FAISS vector databases
│   ├── .env              # (You must create this!) Your secret keys
│   ├── Dockerfile        # Instructions to build the Python app
│   ├── requirements.txt  # Python packages
│   └── streamlit_app.py  # The chatbot web interface
│
├── database/
│   └── Chinook_PostgreSql.sql # The SQL data for the database
│
├── docker-compose.yml    # Main file to run both app and DB
└── README.md             # This file
```