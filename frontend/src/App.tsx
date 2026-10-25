import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Icon, type IconName } from "./components/Icon";
import Explorer from "./pages/Explorer";
import Forecasting from "./pages/Forecasting";
import Inventory from "./pages/Inventory";
import Models from "./pages/Models";
import Monitoring from "./pages/Monitoring";
import Overview from "./pages/Overview";
import Pricing from "./pages/Pricing";
import Trends from "./pages/Trends";

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
}

interface NavGroup {
  group: string;
  items: NavItem[];
}

const NAV: NavGroup[] = [
  {
    group: "Analysis",
    items: [
      { to: "/overview", label: "Overview", icon: "overview" },
      { to: "/trends", label: "Trends", icon: "trends" },
      { to: "/forecasting", label: "Forecasting", icon: "forecast" },
      { to: "/explorer", label: "Explorer", icon: "explorer" },
    ],
  },
  {
    group: "Planning",
    items: [
      { to: "/pricing", label: "Pricing", icon: "pricing" },
      { to: "/inventory", label: "Inventory", icon: "inventory" },
    ],
  },
  {
    group: "Model operations",
    items: [
      { to: "/models", label: "Models", icon: "models" },
      { to: "/monitoring", label: "Monitoring", icon: "monitoring" },
    ],
  },
];

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  // Close the mobile drawer and start each page at the top.
  useEffect(() => {
    setMenuOpen(false);
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <div className="app">
      <header className="topbar">
        <button
          className="icon-btn"
          aria-label="Open navigation"
          aria-expanded={menuOpen}
          aria-controls="sidebar"
          onClick={() => setMenuOpen(true)}
        >
          <Icon name="menu" size={18} />
        </button>
        <span className="topbar-brand">
          <span className="brand-mark" aria-hidden="true">
            E
          </span>
          ElasticIQ
        </span>
      </header>

      <aside
        id="sidebar"
        className={menuOpen ? "sidebar open" : "sidebar"}
        aria-label="Sidebar"
      >
        <button
          className="icon-btn sidebar-close"
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        >
          <Icon name="close" size={18} />
        </button>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            E
          </span>
          <div>
            <div className="brand-name">ElasticIQ</div>
            <div className="brand-sub">Retail demand intelligence</div>
          </div>
        </div>

        <nav aria-label="Main navigation">
          {NAV.map((group) => (
            <div key={group.group}>
              <div className="nav-group">{group.group}</div>
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    isActive ? "nav-link active" : "nav-link"
                  }
                >
                  <span className="nav-icon">
                    <Icon name={item.icon} />
                  </span>
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-foot">
          Observational data. Pricing effects are assumptions, not causality.
        </div>
      </aside>

      {menuOpen ? (
        <button
          className="overlay"
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        />
      ) : null}

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
