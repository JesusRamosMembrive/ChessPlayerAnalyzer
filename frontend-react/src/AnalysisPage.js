import React from 'react';

function AnalysisPage() {
  // Mock data for player history
  const playerHistory = [
    { id: 1, name: 'PlayerAlpha' },
    { id: 2, name: 'PlayerBeta' },
    { id: 3, name: 'PlayerGamma' },
  ];

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4 text-gray-100">Player Analysis</h1>

      <div className="mb-4"> {/* Added margin-bottom for spacing */}
        <label htmlFor="playerName" className="mr-2 text-gray-300">Player Name: </label> {/* Added styling for label */}
        <input type="text" id="playerName" name="playerName" className="bg-gray-700 border border-gray-600 text-white rounded px-3 py-2 focus:outline-none focus:border-blue-500" />
      </div>

      <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ml-2">Start Analysis</button>

      <div className="mt-6"> {/* Increased top margin */}
        <h2 className="text-xl font-semibold mb-3 text-gray-100">Analyzed Players History</h2>
        {playerHistory.length > 0 ? (
          <ul className="list-disc pl-5 mt-4 text-gray-300">
            {playerHistory.map(player => (
              <li key={player.id} className="mb-2">{player.name}</li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-400">No players analyzed yet.</p>
        )}
      </div>
    </div>
  );
}

export default AnalysisPage;
