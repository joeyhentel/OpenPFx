<!-- Improved compatibility of back to top link -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/your_username/OpenPFx">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">OpenPFx</h3>

  <p align="center">
    Open-source toolkit and Streamlit app for generating patient-friendly explanations (PFx) of radiology incidental findings.
    <br />
    <a href="https://github.com/your_username/OpenPFx"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/OpenPFx">View Demo</a>
    ·
    <a href="https://github.com/your_username/OpenPFx/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/your_username/OpenPFx/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#research-notebooks">Research Notebooks</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

**OpenPFx** is a research-driven application designed to generate **patient-friendly explanations (PFx)** of incidental findings in radiology reports.  
It provides multiple prompting strategies (zero-shot, few-shot, multi-few, and agentic workflows) and includes automated readability scoring and ICD-10 labeling.

Core goals:
* Translate clinical findings into clear, accessible explanations for patients.
* Suggest relevant ICD-10 codes using few-shot prompts.
* Evaluate explanation readability using Flesch Reading Ease.
* Provide an easy-to-use **Streamlit app** to browse pre-generated PFx or create new ones.

Pre-generated CSVs are included for reproducibility in `Generated_PFx_CSVs/`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Streamlit][Streamlit]][Streamlit-url]
* [![LangChain][LangChain]][LangChain-url]
* [![AG2][AG2]][AG2-url]
* [![OpenAI][OpenAI]][OpenAI-url]
* [![Pandas][Pandas]][Pandas-url]
* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

* Python 3.9+
* pip (or conda)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username/OpenPFx.git
   cd OpenPFx
Create and activate a virtual environment

sh
Copy code
python -m venv .venv
source .venv/bin/activate   # on Windows use .venv\Scripts\activate
Install dependencies

sh
Copy code
pip install -r requirements.txt
Set environment variables in a .env file

env
Copy code
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o
Run the app

sh
Copy code
streamlit run streamlit.py
<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- USAGE EXAMPLES -->
Usage
OpenPFx provides three main workflows in the Streamlit UI:

Home: introduction, disclaimers, and project overview.

Browse PFx: filter and explore pre-generated CSVs for zero-shot, few-shot, multi-few, and agentic methods.

Generate: create PFx from input findings, view ICD-10 suggestions, and check readability scores.

For more examples, please refer to the Documentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- RESEARCH NOTEBOOKS -->
Research Notebooks
The jh_main/*.ipynb notebooks contain the experiments used in the research project.
They were used to:

Test zero-shot, few-shot, multi-few, and agentic prompting.

Generate PFx at scale for evaluation.

Produce the final CSVs in Generated_PFx_CSVs/.

⚠️ These notebooks document the research process and may not be as polished as the main app. For practical use, run the Streamlit interface instead.

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- ROADMAP -->
Roadmap
 Add Zero-shot / Few-shot / Multi-few PFx

 Add Agentic workflow with AG2

 Add Readability Scoring

 Expand ICD-10 labeling

 Add evaluation metrics dashboard

 Multi-language PFx generation

See the open issues for full list of features and bugs.

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CONTRIBUTING -->
Contributing
Contributions make open source better!
If you have suggestions for improvement, feel free to fork the repo and create a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/MyFeature)

Commit your Changes (git commit -m 'Add MyFeature')

Push to the Branch (git push origin feature/MyFeature)

Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- LICENSE -->
License
Code: Distributed under the Unlicense. See LICENSE.txt.

Generated PFx datasets: Released under CC0 1.0 (see Generated_PFx_CC_LICENSE.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CITATION -->
Citation
If you use this code or data, please cite our related publication:

Paper Title (arXiv / Journal link)

bibtex
Copy code
@article{your_citation_key,
  title={Paper Title},
  author={Your Name and Coauthors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CONTACT -->
Contact
Your Name - @your_twitter - email@example.com

Project Link: https://github.com/your_username/OpenPFx

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- ACKNOWLEDGMENTS -->
Acknowledgments
Choose an Open Source License

Streamlit

LangChain

AG2 / AutoGen

Textstat

Shields.io

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- MARKDOWN LINKS & IMAGES -->
yaml
Copy code

---

👉 Do you want me to also **write a one-line project tagline** (something punchy under the title, like “Democratizing patient-friendly radiology explanations with LLMs”)? It helps the README pop.






You said:
also tell them they can visit streamlit at openpfx.streamlit.app


ChatGPT said:
Perfect — that’s a great addition. I’ll add a live demo link right up top so users know they can try it immediately. Here’s the updated part of the README with the Streamlit cloud app link included:

markdown
Copy code
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/your_username/OpenPFx">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">OpenPFx</h3>

  <p align="center">
    Open-source toolkit and Streamlit app for generating patient-friendly explanations (PFx) of radiology incidental findings.
    <br />
    <a href="https://openpfx.streamlit.app"><strong>🌐 Live Demo »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/OpenPFx">View Repo</a>
    ·
    <a href="https://github.com/your_username/OpenPFx/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/your_username/OpenPFx/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>
That way, the Streamlit Cloud app (openpfx.streamlit.app) is highlighted right under the project name with a bold demo link.

👉 Do you want me to regenerate the entire README with this integrated, so you have the final polished version in one file?







You said:
yes


ChatGPT said:
Got it ✅ — here’s the final polished README.md for your repo, with the live demo link (openpfx.streamlit.app) fully integrated. You can copy this straight into your repo root.

markdown
Copy code
<!-- Improved compatibility of back to top link -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/your_username/OpenPFx">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">OpenPFx</h3>

  <p align="center">
    Open-source toolkit and Streamlit app for generating patient-friendly explanations (PFx) of radiology incidental findings.
    <br />
    <a href="https://openpfx.streamlit.app"><strong>🌐 Live Demo »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/OpenPFx">View Repo</a>
    ·
    <a href="https://github.com/your_username/OpenPFx/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/your_username/OpenPFx/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#research-notebooks">Research Notebooks</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://openpfx.streamlit.app)

**OpenPFx** is a research-driven application designed to generate **patient-friendly explanations (PFx)** of incidental findings in radiology reports.  
It provides multiple prompting strategies (zero-shot, few-shot, multi-few, and agentic workflows) and includes automated readability scoring and ICD-10 labeling.

Core goals:
* Translate clinical findings into clear, accessible explanations for patients.
* Suggest relevant ICD-10 codes using few-shot prompts.
* Evaluate explanation readability using Flesch Reading Ease.
* Provide an easy-to-use **Streamlit app** to browse pre-generated PFx or create new ones.

Pre-generated CSVs are included for reproducibility in `Generated_PFx_CSVs/`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Streamlit][Streamlit]][Streamlit-url]
* [![LangChain][LangChain]][LangChain-url]
* [![AG2][AG2]][AG2-url]
* [![OpenAI][OpenAI]][OpenAI-url]
* [![Pandas][Pandas]][Pandas-url]
* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

* Python 3.9+
* pip (or conda)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username/OpenPFx.git
   cd OpenPFx
Create and activate a virtual environment

sh
Copy code
python -m venv .venv
source .venv/bin/activate   # on Windows use .venv\Scripts\activate
Install dependencies

sh
Copy code
pip install -r requirements.txt
Set environment variables in a .env file

env
Copy code
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o
Run the app

sh
Copy code
streamlit run streamlit.py
<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- USAGE EXAMPLES -->
Usage
You can explore OpenPFx in two ways:

🌐 Live Demo: openpfx.streamlit.app

Local Run: use the installation steps above.

The Streamlit app provides three main workflows:

Home: introduction, disclaimers, and project overview.

Browse PFx: filter and explore pre-generated CSVs for zero-shot, few-shot, multi-few, and agentic methods.

Generate: create PFx from input findings, view ICD-10 suggestions, and check readability scores.

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- RESEARCH NOTEBOOKS -->
Research Notebooks
The jh_main/*.ipynb notebooks contain the experiments used in the research project.
They were used to:

Test zero-shot, few-shot, multi-few, and agentic prompting.

Generate PFx at scale for evaluation.

Produce the final CSVs in Generated_PFx_CSVs/.

⚠️ These notebooks document the research process and may not be as polished as the main app. For practical use, run the Streamlit interface instead.

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- ROADMAP -->
Roadmap
 Add Zero-shot / Few-shot / Multi-few PFx

 Add Agentic workflow with AG2

 Add Readability Scoring

 Expand ICD-10 labeling

 Add evaluation metrics dashboard

 Multi-language PFx generation

See the open issues for full list of features and bugs.

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CONTRIBUTING -->
Contributing
Contributions make open source better!
If you have suggestions for improvement, feel free to fork the repo and create a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/MyFeature)

Commit your Changes (git commit -m 'Add MyFeature')

Push to the Branch (git push origin feature/MyFeature)

Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- LICENSE -->
License
Code: Distributed under the Unlicense. See LICENSE.txt.

Generated PFx datasets: Released under CC0 1.0 (see Generated_PFx_CC_LICENSE.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CITATION -->
Citation
If you use this code or data, please cite our related publication:

Paper Title (arXiv / Journal link)

bibtex
Copy code
@article{your_citation_key,
  title={Paper Title},
  author={Your Name and Coauthors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- CONTACT -->
Contact
Your Name - @your_twitter - email@example.com

Project Link: https://github.com/your_username/OpenPFx

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- ACKNOWLEDGMENTS -->
Acknowledgments
Choose an Open Source License

Streamlit

LangChain

AG2 / AutoGen

Textstat

Shields.io

<p align="right">(<a href="#readme-top">back to top</a>)</p> <!-- MARKDOWN LINKS & IMAGES -->
