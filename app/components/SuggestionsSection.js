'use client';

import React from 'react';
import { Card, CardHeader, CardBody, Button, Chip } from "@nextui-org/react";

const SuggestionsSection = ({ title, suggestions, options, onReset, onAdd, onRemove }) => {
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
                            onClick={() => onAdd(option.value)}
                            onClose={() => onRemove(option, false)} // Ensure the delete button works
                            classNames={{
                                base: "flex items-center px-3 py-1 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-full shadow transition  duration-300 ease-in-out transform hover:-translate-y-1 hover:scale-105 cursor-pointer",
                                closeButton: "ml-2 text-xl bg-white rounded-full p-1 hover:bg-red-500 transition duration-300 transform hover:scale-110",
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
                    <h3 className="text-lg font-semibold mt-4">{title}</h3>
                    <Button auto flat onPress={onReset}>Reset</Button>
                </CardHeader>
                <CardBody className="mt-4">
                    {renderTags()}
                </CardBody>
            </Card>
        </div>
    );
};

export default SuggestionsSection;
