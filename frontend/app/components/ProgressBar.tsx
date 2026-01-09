import styles from "./ProgressBar.module.css";

interface ProgressBarProps {
    current: number;
    total: number;
}

export default function ProgressBar({ current, total }: ProgressBarProps) {
    const percentage = Math.min(100, Math.max(0, (current / total) * 100));

    return (
        <div>
            <div className={styles.container}>
                <div
                    className={styles.bar}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            <div className={styles.text}>
                Processing {current} of {total} images...
            </div>
        </div>
    );
}
