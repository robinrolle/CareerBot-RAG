import React from 'react';
import { Card, CardHeader, CardBody, Chip } from "@nextui-org/react";

const SuggestionsSection = ({ suggestions, options, onSelect, onSuggest }) => {
    const renderTags = () => {
        if (suggestions.length === 0) {
            return <div className="text-gray-500 mb-4">No suggestions</div>;
        }

        return (
            <div className="flex gap-2 flex-wrap mb-4 ">
                {suggestions.map(value => {
                    const option = options.find(opt => opt.value === value);
                    return option ? (
                        <Chip
                            key={option.value}
                            onClick={() => onSelect(option)}
                            classNames={{
                                base: "flex items-center px-3 py-1 text-m font-medium text-gray-800 bg-white border border-gray-300 rounded-full shadow cursor-pointer transition hover:bg-green-500  duration-300 ease-in-out transform hover:-translate-y-1 hover:scale-105",
                            }}
                        >
                            {option.label}
                        </Chip>
                    ) : null;
                })}
            </div>
        );
    };

    return (
        <div className="suggestions-container bg-white shadow-lg rounded-lg px-5 py-4">
            <Card className='px-1'>
                <CardHeader className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold mt-4">Suggestions</h3>
                    <button
                        className="p-2 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-md shadow  hover:bg-red-500 transition duration-300 transform hover:scale-110"
                        onClick={onSuggest}
                    >
                        Suggest
                    </button>
                </CardHeader>
                <CardBody className="mt-4">
                    <div className="selections-container">
                        <div>{renderTags()}</div>
                    </div>
                </CardBody>
            </Card>
        </div>
    );
};

export default SuggestionsSection;
