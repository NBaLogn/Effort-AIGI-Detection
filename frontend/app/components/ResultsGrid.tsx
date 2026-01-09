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
    hasSummary?: boolean;
}

export default function ResultsGrid({ results, hasSummary = false }: ResultsGridProps) {
    if (results.length === 0) return null;

    const marginTop = hasSummary ? '1.5rem' : '3rem';

    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '2rem',
            marginTop
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
