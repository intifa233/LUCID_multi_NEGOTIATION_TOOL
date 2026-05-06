# Quick Start Guide — LUCID Toolkit — GPT + Qualtrics Backend 

Welcome to the backend server powering the [LUCID Toolkit](http://lucidresearch.io) — a research infrastructure for running controlled GPT-based chatbot interactions directly inside **Qualtrics** surveys.

This code (Flask backend) serves as a **secure proxy to the OpenAI API** — allowing each researcher or research team to create their own isolated, privacy-preserving implementation of the LUCID system.

---

## ⚠️ Why Is a Separate Server Needed?

It is **not safe to embed an OpenAI API key directly** in Qualtrics — whether in JavaScript, Survey Flow, or an HTML element. Survey participants (or anyone inspecting the page) could easily view and copy your key.

The LUCID backend solves this by securely handling GPT requests *on your own private server*. The API key is never visible to participants. And the best part? 
1. You only need to do this **once**
2. The service we recommend, Vercel, has a **free tier** which is sufficient for most researchers' use. 

---

## What You'll Need Before Getting Started

You only need two accounts to get started:

1. **An OpenAI API Key**  
   - Sign up at [platform.openai.com](https://platform.openai.com/signup)  
   - Then follow [this beginner's guide](https://questionableresearch.ai/2025/02/27/how-to-sign-up-for-openai-and-get-an-api-key-for-beginners/) if you're new to OpenAI's API 
   - You'll paste this key into your backend deployment (not into Qualtrics!)

2. **A GitHub Account**  
   - Sign up at [github.com/join](https://github.com/join)  
   - Used to deploy the backend via Vercel in just a few clicks.
  
3. (Optional) **A printable version of our detailed setup guide containing all steps with screenshots**
📄 [Guide (.docx)](https://github.com/amgarv/LUCID_TOOL_BACKEND/raw/main/LUCIDToolkitUserGuide.docx) 

If you have all this, you're ready to get started. 

## Step 1: Deploy YOUR LUCID Backend

To get started, click the following button:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Famgarv%2FLUCID_TOOL_BACKEND&project-name=lucid-tool-backend&repository-name=my-lucid-backend-code&env=OPENAI_API_KEY&envDescription=INSTRUCTIONS%3A%20For%20OPENAI_API_KEY%20input%20your%20OpenAI%20API%20key%20(available%20at%20%22Learn%20more%22%20link%20below).&envLink=https%3A%2F%2Fplatform.openai.com%2Faccount%2Fapi-keys)

You'll be taken to Vercel.

## 1.1 Deployment Steps (Using the Deploy Button)

1.  **Click the "Deploy with Vercel" button** above.
2.  **Connect Git Provider:** When prompted, connect your GitHub account.
3.  **Create Project:**
    * Vercel will suggest a **Project Name**. You can keep it or change it (e.g., `my-lucid-backend-code`). This name will determine your default backend deployment URL that will be used by Qualtrics, so use a name you are comfortable with others (such a reviewers) potentially seeing.
    * Click "Create". Vercel will create a copy of the LUCID code from the repository into your account.
4.  **Configure Project:** This is crucial. After the project is created, Vercel will prompt you for:
    * **`OPENAI_API_KEY`**:
        * **Purpose:** Allows LUCID to access OpenAI ChatGPT models.
        * **Action:** Go to [OpenAI API Keys](https://platform.openai.com/api-keys), create a new "Secret key", copy it, and paste it into the `OPENAI_API_KEY` value field in Vercel. 
        * **Optional but Recommended:** For security, it is a good practice to disable your secret key on the OpenAI platform when you are not actively collecting data with a LUCID Qualtrics study. You can then re-enable the key on the OpenAI platform when you field a study.
6.  **Deploy:** Click the "Deploy" button in Vercel.
7.  **Wait:** Vercel will build and deploy your backend. This usually takes 1-2 minutes.

## 1.2 Getting Your Backend URL for Qualtrics

**Visit Your Deployment:** Once Vercel shows the deployment is complete ("Congratulations!"), click the image displayed that says in blue text "LUCID Backend Succesfully Deployed!".

## 1.3 Save your Backend URL

**Copy the URL:** The resulting page you visit should display "LUCID Backend Successfully Deployed!". It will clearly show the **exact URL needed for Qualtrics**. Click the "Copy Backend URL" button (e.g., `https://your-project-name.vercel.app/lucid`). Keep this address handy for the Qualtrics setup.

---

# 🎉 You are now ready for Step 2! 🥳

Download your .QSF file from this repository:

LUCID_Qualtrics_Template_1_-One_Group_Design.qsf - A single condition design (e.g., for conducting an interview using the same chatbot for all participants)

LUCID_Qualtrics_Template_2_-Two_Group_Between_Subjects_Experiment.qsf - A 2-Group between subjects design (e.g., for conducting an experiment that manipulates chatbot behavior) 

---

## References

Garvey, Aaron M. and Simon J. Blanchard, (2025) "Generative AI as a Research Confederate: The LUCID Methodological Framework and Toolkit for Human-AI Interactions Research," MSI Working Paper. [paper@ssrn](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5256150) 

Blanchard, S. J., Duani, N., Garvey, A. M., Netzer, O., & Tae Oh, T. (2025). New Tools, New Rules: A Practical Guide to Effective and Responsible Generative AI Use for Surveys and Experiments in Research. Journal of Marketing, 89(6), 119-139. [paper@Journal of Marketing](https://doi.org/10.1177/00222429251349882)

The code contents of this git repository are available for non-commercial use under Creative Commons BY-NC-SA (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).
