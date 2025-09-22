# OpenPFx
## Making medical reports understandable.
OpenPFx is an open-source program that uses generative AI to generate patient-friendly explanations (PFx) of medical findings. 

**[Visit Website](https://openpfx.streamlit.app)**

## Key Features 

### 📊 Advanced Stats:

**ICD-10 Codes**: standard medical codes used worldwide to label diseases and health conditions

> DISCLAIMER: ICD-10 codes provided by this tool are approximations for informational use and may be inaccurate. Always confirm final codes with a licensed clinician or certified medical coder.


**Readability**: Flesch Reading Ease Score (FRES)
> <img src="./images/FRES_Guide.png" alt="FRES Guide" width="400"/>


**Accuracy**: measured through mutliple ICD-10 checks
> When the LLM generates a PFx, it also assigns an ICD-10 code to its own response. A separate agent then reads that PFx and assigns a code as well. The first three digits of both predictions are compared to the first three digits of the true ICD-10 code. Each correct match earns a point, and the two points are averaged.

### 📝 Pre-Generated PFx: 

On website, navigate to Browse PFx page. Select desired workflow (see workflow explanations below) and finding(s).

[Acesss Through Raw CSV Files](./Generated_PFx_CSVs/)


### 🖥️ Generate Your Own PFx:

On website, navigate to Generate page. Input finding, model, desired grade level, workflow (see explanation below), and ICD-10 Code (optional)
>ICD-10 code will be autofilled if you don't input one
>FRES Score + Accuracy available


## Workflows Explained
OpenPFx offers four approaches for generating explanations
[See All Prompts Here](./Generated_PFx_CSVs/)

### Zero-shot Prompting 
No context such as a template or an example letter is provided, requiring the LLM to rely solely on its previous knowledge. 

### Few-shot Prompting
Provides context such as sample responses or templates

### Agentic 
Structured process that enables AI models, acting as "agents," to iteratively refine their outputs based on feedback and contextual adjustments <img width="468" height="42" alt="image" 




