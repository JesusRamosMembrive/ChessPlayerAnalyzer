import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import AnalysisPage from './AnalysisPage';
import DashboardPage from './DashboardPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center">
        <nav className="p-4 shadow-md w-full bg-gray-800">
          <ul className="flex justify-center space-x-4">
            <li>
              <Link to="/" className="hover:text-gray-400">Analysis</Link>
            </li>
            <li>
              <Link to="/dashboard" className="hover:text-gray-400">Dashboard</Link>
            </li>
          </ul>
        </nav>

        <hr className="border-gray-700 w-full" />

        <div className="p-4 w-full max-w-4xl">
          <Routes>
            <Route path="/" element={<AnalysisPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
