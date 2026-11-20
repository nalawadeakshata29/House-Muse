# 🏡 House Muse – AI-Based Home Décor and Interior Designer

**House Muse** is an AI-powered interior design and room redesign web application that helps users visualize how their existing rooms could look in different interior design styles.

Users can upload a photograph of their room, select a preferred interior style, and generate an AI-redesigned visualization. The system combines Generative AI, Computer Vision, and web technologies to make interior design exploration easier and more accessible.

## ✨ Features

- **Room Image Upload:** Upload an image of an existing room for redesign.
- **Multiple Interior Styles:** Choose from nine interior design styles:
  - Coastal
  - Industrial
  - Contemporary Indian
  - Vintage Indian
  - Scandinavian
  - Eclectic
  - Shabby Chic
  - Mid-Century Modern
  - Victorian
- **AI-Based Room Redesign:** Generate redesigned room visualizations using Stable Diffusion XL (SDXL).
- **Object Detection:** Identify furniture and other room objects using YOLOv8.
- **Colour Palette Processing:** Analyze room colours to support the design process.
- **User Authentication:** Secure user access using Django Knox.
- **Project Management:** Store and manage saved design projects.
- **Design Refinement:** Support further modifications to generated designs.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| UI Styling | Tailwind CSS |
| API Communication | Axios |
| Backend | Python, Django, Django REST Framework |
| ML Service | Flask |
| Image Generation | Stable Diffusion XL (SDXL) |
| Structural Guidance | ControlNet |
| Style Customization | LoRA |
| Object Detection | YOLOv8 |
| Image Processing | OpenCV, PyTorch |
| Database | PostgreSQL / SQLite |
| Authentication | Django Knox |
| Image Generation Hosting | Hugging Face Spaces, ZeroGPU |

## 🏗️ System Architecture

House Muse follows a modular, service-oriented architecture. The frontend, backend, computer vision service, and image generation service operate as separate components and communicate through APIs.

### Application Workflow

1. The user uploads a room image and provides room details and design preferences.
2. The React frontend sends the request to the Django REST API.
3. Django handles authentication, request validation, and project management.
4. The Flask service processes the room image using computer vision techniques.
5. YOLOv8 detects objects, while colour analysis extracts visual information.
6. The image generation pipeline uses SDXL, ControlNet, and style-specific LoRA models to generate a redesigned room.
7. The redesigned image is returned through the application services and displayed to the user.
8. Users can save their design projects for future reference.

## 📂 Project Structure

The application is divided into independent modules:

```text
House-Muse/
│
├── Frontend/              # React-based user interface
│
├── Backend/               # Django REST API and application logic
│
├── ML-Service/             # Flask-based image processing service
│
├── Image-Generation/       # AI image generation integration
│
├── datasets/               # Image and style-related resources
│
├── README.md
└── requirements.txt
```

*Note: The directory names above are illustrative. Adjust them to match the actual folders in your repository.*

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.10+
- Node.js and npm
- Git
- PostgreSQL (if using PostgreSQL)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/House-Muse.git
cd House-Muse
```

**2. Set up the backend**

```bash
cd Backend
python -m venv venv
```

Activate the virtual environment.

Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**3. Set up the frontend**

Open another terminal:

```bash
cd Frontend
npm install
npm run dev
```

**4. Run the backend**

From the backend directory:

```bash
python manage.py runserver
```

*The exact commands and environment variables may vary depending on the actual repository configuration. The Flask service and image-generation service must also be configured and started separately.*

## 🎯 Project Objective

The main objective of House Muse is to combine artificial intelligence and computer vision to simplify early-stage interior design exploration.

Instead of manually imagining how a room might look after redesigning, users can explore different styles through AI-generated visualizations before making real-world design decisions.

## ⚠️ Limitations

- AI-generated images are visual design references, not construction-ready blueprints.
- Furniture placement and room geometry may not always remain perfectly consistent.
- Generated results depend on the image-generation model and its configuration.
- Budget estimates and décor suggestions should be treated as guidance rather than guaranteed costs.

## 🔮 Future Scope

- Improve room geometry and object-position preservation.
- Expand the collection of interior design styles.
- Improve personalized design recommendations.
- Enhance budget-based furniture and décor suggestions.
- Improve the speed and reliability of image generation.

## 👩‍💻 Contributors

Developed as an academic project by:

- Akshata Nalawade
- Alisha Sawant
- Arnav Tripathi

**Department:** Electronics and Computer Science  
**Institution:** Vidyalankar Institute of Technology  
**Academic Year:** 2025–2026

---

⭐ If you find this project interesting, feel free to explore the repository!
