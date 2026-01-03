# Sage Analytics Dashboard

Real-time analytics dashboard for the Sage Discord support bot with stunning 3D neon background.

## Features

- ✨ **3D Neon Raymarcher Background** - Animated iridescent cubes
- 📊 **Real-time Metrics** - Total queries, satisfaction rate, escalation rate
- 📈 **Weekly Trend Charts** - Visual activity trends
- 🕐 **Time Range Filter** - 7-day or 30-day views
- 🎨 **Glassmorphism UI** - Polished, modern design
- ⚡ **Fast & Responsive** - Optimized for all devices

## Tech Stack

- Next.js 15 + React 19
- TypeScript
- Tailwind CSS
- Three.js + React Three Fiber
- Supabase (database + auth)
- Recharts

## Setup

### 1. Install Dependencies

```bash
cd dashboard
npm install
```

### 2. Configure Environment Variables

Edit `.env.local` and add your Supabase keys:

```env
NEXT_PUBLIC_SUPABASE_URL=https://oiaycqwjjxuqsjhghaiy.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
```

**Get your anon key:**
1. Go to Supabase Dashboard → Project Settings → API
2. Copy the `anon` key (public key, NOT service_role)

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deployment

### Deploy to Vercel

1. Push code to GitHub:
```bash
git add .
git commit -m "Add Sage analytics dashboard"
git push origin main
```

2. Go to [Vercel](https://vercel.com)
3. Import your GitHub repository
4. Add environment variables:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
5. Deploy!

## Dashboard Metrics

| Metric | Description |
|--------|-------------|
| **Total Queries** | Number of questions answered by the bot |
| **Satisfaction Rate** | % of users who clicked "Got it, thanks!" |
| **Escalation Rate** | % of users who clicked "Tag the crew" |
| **Weekly Trend** | Visual chart showing activity over time |

## Design

- **Audience**: Program team, Mentors, Leadership
- **Style**: Dark, tech-forward with green accents
- **Typography**: Inter (clean, professional)
- **Theme**: Inspired by "Sage" - wise, natural, tech-forward

## Project Structure

```
dashboard/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx         # Main dashboard
│   │   └── globals.css
│   ├── components/
│   │   └── ui/
│   │       └── neon-raymarcher.tsx  # 3D background
│   └── lib/
│       ├── supabase.ts      # Supabase client
│       └── utils.ts         # Utility functions
├── package.json
├── tsconfig.json
└── tailwind.config.ts
```

## Support

For issues or questions, contact the 100xEngineers team.
