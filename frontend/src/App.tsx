import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Trends from "./pages/Trends";
import Forecasting from "./pages/Forecasting";
import Explorer from "./pages/Explorer";
import Pricing from "./pages/Pricing";
import Inventory from "./pages/Inventory";
import Models from "./pages/Models";
import Monitoring from "./pages/Monitoring";

const NAV = [
  { to: "/overview", label: "Overview", icon: "▣" },
  { to: "/trends", label: "Sales trends", icon: "∿" },
  { to: "/forecasting", label: "Forecasting", icon: "⌁" },
  { to: "/explorer", label: "Products & stores", icon: "☰" },
  { to: "/pricing", label: "Pricing what-if", icon: "$" },
  { to: "/inventory", label: "Inventory", icon: "▤" },
  { to: "/models", label: "Model performance", icon: "◈" },
  { to: "/monitoring", label: "Monitoring", icon: "◉" },
];

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">E</span>
          <div>
            <div className="brand-name">ElasticIQ</div>
            <div className="brand-sub">M5 retail intelligence</div>
          </div>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          Observational data. Pricing effects are assumptions, not causality.
        </div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/trends" element={<Trends />} />
          <Route path="/forecasting" element={<Forecasting />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/inventory" element={<Inventory />} />
          <Route path="/models" element={<Models />} />
          <Route path="/monitoring" element={<Monitoring />} />
        </Routes>
      </main>
    </div>
  );
}
