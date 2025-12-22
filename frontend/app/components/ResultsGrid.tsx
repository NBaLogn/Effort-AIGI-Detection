import React from 'react';
import ResultCard from './ResultCard';

interface Result {
    filename: string;
    originalImage: string;
    gradCamImage?: string;
    label: string;
    score: number;
}

interface ResultsGridProps {
    results: Result[];
}

export default function ResultsGrid({ results }: ResultsGridProps) {
    if (results.length === 0) return null;

    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '2rem',
            marginTop: '3rem'
        }}>
            {results.map((result, index) => (
                <ResultCard
                    key={index}
                    {...result}
                />
            ))}
        </div>
    );
}
