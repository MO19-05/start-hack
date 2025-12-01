# START Hackathon - One-Ware Challenge


## Project vision

An end-to-end computer vision solution that enables real-time monitoring of physical spaces, providing actionable insights for occupancy management and facility cleanliness. By leveraging One-Ware's integrated platform, we demonstrate how to build, train, and deploy production-ready AI models in 24 hours.

Running them on different hardware devices and comparing their performance. 

## Core Capabilities

We built two computer vision models trained using One-Ware:

**1. Cleanliness Classification:**

Automatically identifies when spaces require cleaning by detecting visual indicators of uncleanliness—clutter, debris, spills, or general disorder—triggering proactive facility management.

**2. People counting & occupancy detection:**

Accurately counts and tracks the number of occupants in real-time, enabling dynamic utilization insights and occupancy analytics.

## Problem

Modern workplaces and shared spaces face persistent challenges:

- Facility managers lack real-time visibility into space conditions and occupancy, leading to inefficient cleaning schedules and reactive rather than proactive maintenance
- Shared workspace operators struggle to maintain premium experiences for clients, resulting in lost revenue and damaged reputation when spaces aren't appropriately maintained
- Cleaning service providers operate on fixed schedules without demand intelligence, leading to either under-serviced spaces or wasted resources
- Employees and visitors experience discomfort in poorly maintained or overcrowded environments, affecting productivity and satisfaction

## Target Applications & Use Cases

Corporate Office Buildings

Smart Occupancy Management: Track meeting room utilization in real-time to optimize desk-sharing policies and space allocation
Predictive Facility Maintenance: Schedule cleaning based on actual occupancy patterns rather than fixed time slots
Employee Experience: Ensure conference rooms, break areas, and collaborative spaces are clean and available when needed

Coworking Spaces & Shared Workplaces

Premium Member Experience: Maintain spotless shared areas, common zones, and facilities to justify premium pricing and attract clients
Dynamic Cleaning Dispatch: Alert cleaning staff to high-traffic areas that need immediate attention during business hours
Occupancy-Based Pricing: Implement dynamic pricing models based on real-time space availability and demand

Cleaning Service Companies

Data-Driven Operations: Transition from time-based to condition-based scheduling, improving efficiency and customer satisfaction
Performance Verification: Provide clients with objective visual evidence that cleaning standards have been met
Route Optimization: Prioritize spaces by need rather than arbitrary schedules, reducing operational costs


## Getting Started

### Prerequisites

- Node.js & npm installed on your machine.

### Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd start-hack-one-ware
   ```

2. Install dependencies:
   ```sh
   npm install
   # or
   bun install
   ```

3. Configure Environment Variables:
   Create a `.env` file in the root directory with your Supabase credentials:
   ```env
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_anon_key
   ```

4. Start the development server:
   ```sh
   npm run dev
   ```

### Backend & Edge Functions

The project uses Supabase Edge Functions for features like email alerts.

- **send-cleaning-alert**: Sends an email notification when a room is marked for cleaning.
  - Requires `RESEND_API_KEY` to be set in Supabase secrets.
  - Uses [Resend](https://resend.com) for email delivery.

## Technical Stack

- **AI/ML Platform**: One-Ware
- **Framework**: React + Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Backend**: Supabase (Database & Edge Functions)
- **State Management**: TanStack Query
- **Email Service**: Resend

## Team
Ska, Jasser, Moritz
