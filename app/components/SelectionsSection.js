import React from 'react';
import { Card, CardHeader, CardBody, Chip } from "@nextui-org/react";
import SearchBar from './SearchBar';

const SelectionsSection = ({ title, selections, options, placeholder, onChange, onReset, onRemove }) => {
    const getRelevanceColor = (relevance) => {
        switch (relevance) {
            case 'HIGH':
                return 'bg-green-500';
            case 'MEDIUM':
                return 'bg-yellow-500';
            case 'LOW':
                return 'bg-red-500';
            default:
                return 'bg-gray-500';
        }
    };

    const renderColorLegend = () => (
        <div className="flex items-center space-x-4 mb-2">
            <span className="text-sm font-medium">Relevance:</span>
            {[
                { color: 'bg-green-500', label: 'High' },
                { color: 'bg-yellow-500', label: 'Medium' },
                { color: 'bg-red-500', label: 'Low' },
            ].map(({ color, label }) => (
                <div key={label} className="flex items-center">
                    <span className={`inline-block w-2 h-2 rounded-full mr-1 ${color}`}></span>
                    <span className="text-xs">{label}</span>
                </div>
            ))}
        </div>
    );

    const renderTags = () => {
        if (selections.length === 0) {
            return <div className="text-gray-500 mb-4">No selections</div>;
        }

        return (
            <div className="flex gap-2 flex-wrap mb-4 ">
                {selections.map(selection => {
                    const option = options.find(opt => opt.value === selection.id);
                    const dotColor = getRelevanceColor(selection.relevance);
                    return (
                        <Chip
                            key={selection.id}
                            onClose={() => onRemove(selection.id)}
                            classNames={{
                                base: "flex items-center px-3 py-1 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-full shadow cursor-default",
                                closeButton: "ml-2 text-xl bg-white rounded-full p-1 hover:bg-red-500 transition duration-300 transform hover:scale-110",
                            }}
                        >
                            <span className={`inline-block w-2 h-2 rounded-full mr-2 ${dotColor}`}></span>
                            {selection.item}
                        </Chip>
                    );
                })}
            </div>
        );
    };

    return (
        <div className="suggestions-container bg-white shadow-lg rounded-lg px-5 py-4">
            <Card className='px-1'>
                <CardHeader className="flex flex-col items-start">
                    <div className="flex justify-between items-center w-full">
                        <h3 className="text-lg font-semibold mt-4">{title}</h3>
                        <button
                            className="p-2 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-md shadow hover:bg-red-500 transition duration-300 transform hover:scale-110"
                            onClick={onReset}
                        >
                            Reset
                        </button>
                    </div>
                    {renderColorLegend()}
                </CardHeader>
                <CardBody className="mt-4">
                    <div className="selections-container">
                        <SearchBar
                            options={options}
                            placeholder={placeholder}
                            onChange={(newValues) => onChange(newValues)}
                            value={selections.map(s => s.id)}
                        />
                        <div>{renderTags()}</div>
                    </div>
                </CardBody>
            </Card>
        </div>
    );
};

export default SelectionsSection;