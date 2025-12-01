# Frontend — Local development

This frontend is a Vite + React + TypeScript app. To run it locally:

Prerequisites
- Node.js 18+ (we recommend using nvm or your package manager)
- npm, pnpm or yarn

Quick start (using `npm`)

```bash
cd frontend
npm install
# Create a local .env file (see .env.example)
# For local development, point the frontend to the backend API:
# VITE_BACKEND_URL=http://localhost:8000
echo "VITE_BACKEND_URL=http://localhost:8000" > .env
npm run dev
```

Build for production

```bash
npm run build
npm run preview
```

Notes
- The app reads the backend URL from `import.meta.env.VITE_BACKEND_URL`. If not set it defaults to `http://localhost:8000`.
- If you want to use a different port or host for the backend, update `.env` accordingly and restart the dev server.
# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/fb1e0064-31c5-4fb5-a082-350ab4da3bf1

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/fb1e0064-31c5-4fb5-a082-350ab4da3bf1) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/fb1e0064-31c5-4fb5-a082-350ab4da3bf1) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
