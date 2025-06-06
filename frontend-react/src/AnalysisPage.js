import React from 'react';

function AnalysisPage() {
  // Mock data for player history
  const playerHistory = [
    { id: 1, name: 'PlayerAlpha' },
    { id: 2, name: 'PlayerBeta' },
    { id: 3, name: 'PlayerGamma' },
  ];

  return (
    <div>
      <h1>Player Analysis</h1>

      <div>
        <label htmlFor="playerName">Player Name: </label>
        <input type="text" id="playerName" name="playerName" />
      </div>

      <button style={{ marginTop: '10px' }}>Start Analysis</button>

      <div style={{ marginTop: '20px' }}>
        <h2>Analyzed Players History</h2>
        {playerHistory.length > 0 ? (
          <ul>
            {playerHistory.map(player => (
              <li key={player.id}>{player.name}</li>
            ))}
          </ul>
        ) : (
          <p>No players analyzed yet.</p>
        )}
      </div>
    </div>
  );
}

export default AnalysisPage;
