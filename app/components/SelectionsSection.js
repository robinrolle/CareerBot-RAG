'use client';

import React from 'react';
import { Card, CardHeader, CardBody, Button, Chip } from "@nextui-org/react";
import SearchBar from './SearchBar';

const SelectionsSection = ({ title, selections, options, placeholder, onChange, onRemove }) => {
    const renderTags = () => {
        if (selections.length === 0) {
            return <div className="text-gray-500 mb-4">No selections</div>;
        }

        return (
            <div className="flex gap-2 flex-wrap mb-4 ">
                {selections.map(value => {
                    const option = options.find(opt => opt.value === value);
                    return option ? (
                        <Chip
                            key={option.value}
                            onClose={() => onRemove(option, true)}
                            classNames={{
                                base: "flex items-center px-3 py-1 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-full shadow cursor-defeault",
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
                </CardHeader>
                <CardBody className="mt-4">
                    <div className="selections-container">
                        <SearchBar
                            options={options}
                            placeholder={placeholder}
                            onChange={onChange}
                            value={selections}
                        />
                        <div>{renderTags()}</div>
                    </div>
                </CardBody>
            </Card>
        </div>
    );
};

export default SelectionsSection;
