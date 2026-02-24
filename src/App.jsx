import { useState, useEffect } from 'react';
import { CipherLanding } from './prototypes/CipherLanding';
import { DevelopersPage } from './prototypes/DevelopersPage';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  useEffect(() => {
    // Handle hash-based routing
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1) || 'home';
      setCurrentPage(hash);
    };

    // Set initial page
    handleHashChange();

    // Listen for hash changes
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Render page based on current hash
  switch (currentPage) {
    case 'developers':
      return <DevelopersPage />;
    default:
      return <CipherLanding />;
  }
}

export default App;
